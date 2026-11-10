"""AST-based line-to-symbol mapping service for TestWeaver.

Week 3, Phase 2 — Stage 2:
Maps changed line numbers from parsed Git diffs (ParsedDiff / FileDiff) to
their enclosing Python code symbols (functions, methods, classes, module-level).
Compares base and target Git revisions to classify symbol changes as:
  - ADDED:    Symbol exists only in target revision and has added lines.
  - DELETED:  Symbol exists only in base revision and has deleted lines.
  - MODIFIED: Symbol exists in both revisions and has changed lines.

Symbol identity is defined by the tuple:
  (fully_qualified_name, symbol_type)
preventing symbols of different types (e.g. function replaced by class) from
being merged as modified.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from services.diff_analyzer import FileChangeType, FileDiff, ParsedDiff
from services.git_service import (
    GitServiceError,
    get_file_content_at_revision,
    validate_repository,
    validate_revision,
)


class SymbolType(str, Enum):
    """Classification of Python code symbols."""

    FUNCTION = "FUNCTION"
    ASYNC_FUNCTION = "ASYNC_FUNCTION"
    METHOD = "METHOD"
    ASYNC_METHOD = "ASYNC_METHOD"
    CLASS = "CLASS"
    MODULE = "MODULE"


class ChangeCategory(str, Enum):
    """Classification of symbol-level modifications between revisions."""

    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


@dataclass
class CodeSymbol:
    """Represents a discovered Python code symbol with line boundaries."""

    name: str                           # Short symbol name (e.g. "add")
    fully_qualified_name: str           # Qualified name (e.g. "Calculator.add")
    symbol_type: SymbolType
    line_start: int                     # 1-based, inclusive, decorator-aware
    line_end: int                       # 1-based, inclusive
    parent_name: Optional[str] = None   # Immediate parent name (e.g. "Calculator")


@dataclass
class ParseResult:
    """Result of parsing source code into symbols with error capture."""

    symbols: list[CodeSymbol] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ChangedSymbol:
    """Represents a changed symbol with attached diff evidence."""

    symbol_name: str
    fully_qualified_name: str
    symbol_type: SymbolType
    change_category: ChangeCategory
    line_start: int                     # Target rev for ADDED/MODIFIED; base rev for DELETED
    line_end: int
    changed_lines: list[int] = field(default_factory=list)  # Sorted unique changed lines


@dataclass
class FileSymbolChanges:
    """Aggregated symbol-level change results for a single file."""

    file_path: str
    old_path: Optional[str]
    new_path: Optional[str]
    change_type: FileChangeType
    changed_symbols: list[ChangedSymbol] = field(default_factory=list)
    base_parse_success: bool = True
    base_parse_error: Optional[str] = None
    target_parse_success: bool = True
    target_parse_error: Optional[str] = None


MODULE_KEY: tuple[str, SymbolType] = ("<module>", SymbolType.MODULE)


def get_node_start_line(node: ast.AST) -> int:
    """Return the 1-based start line of an AST node, including any decorators."""
    start = getattr(node, "lineno", 1)
    decorator_list = getattr(node, "decorator_list", None)
    if decorator_list:
        decorator_lines = [
            d.lineno for d in decorator_list if hasattr(d, "lineno")
        ]
        if decorator_lines:
            return min(start, min(decorator_lines))
    return start


def get_node_end_line(node: ast.AST, default: int = 1) -> int:
    """Return the 1-based end line of an AST node, using end_lineno where available."""
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    max_line = getattr(node, "lineno", default)
    for child in ast.walk(node):
        child_end = getattr(child, "end_lineno", getattr(child, "lineno", None))
        if child_end is not None and child_end > max_line:
            max_line = child_end
    return max_line


def parse_source_symbols(source_code: str) -> ParseResult:
    """Parse Python source code into a list of CodeSymbols using ast.

    Captures functions, async functions, classes, methods, and nested functions.
    Handles syntax errors gracefully without raising exceptions.
    """
    if not source_code:
        return ParseResult(symbols=[], success=True)

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return ParseResult(
            symbols=[],
            success=False,
            error_message=f"SyntaxError: {exc.msg} at line {exc.lineno}",
        )
    except Exception as exc:
        return ParseResult(
            symbols=[],
            success=False,
            error_message=f"AST parsing error: {exc}",
        )

    symbols: list[CodeSymbol] = []

    def _visit_nodes(nodes: list[ast.AST], parent_fqn: Optional[str], parent_is_class: bool):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                fqn = f"{parent_fqn}.{name}" if parent_fqn else name
                start_line = get_node_start_line(node)
                end_line = get_node_end_line(node, start_line)

                if isinstance(node, ast.ClassDef):
                    sym_type = SymbolType.CLASS
                elif isinstance(node, ast.AsyncFunctionDef):
                    sym_type = SymbolType.ASYNC_METHOD if parent_is_class else SymbolType.ASYNC_FUNCTION
                else:  # ast.FunctionDef
                    sym_type = SymbolType.METHOD if parent_is_class else SymbolType.FUNCTION

                symbols.append(
                    CodeSymbol(
                        name=name,
                        fully_qualified_name=fqn,
                        symbol_type=sym_type,
                        line_start=start_line,
                        line_end=end_line,
                        parent_name=parent_fqn,
                    )
                )

                # Recursively visit children in class or function body
                is_class = isinstance(node, ast.ClassDef)
                _visit_nodes(node.body, parent_fqn=fqn, parent_is_class=is_class)

            elif hasattr(node, "body") and isinstance(node.body, list):
                # Compound statements: if, while, for, with, try, match
                _visit_nodes(node.body, parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                for orelse in getattr(node, "orelse", []):
                    if isinstance(orelse, list):
                        _visit_nodes(orelse, parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                    elif isinstance(orelse, ast.AST):
                        _visit_nodes([orelse], parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                for handler in getattr(node, "handlers", []):
                    if hasattr(handler, "body") and isinstance(handler.body, list):
                        _visit_nodes(handler.body, parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                for final in getattr(node, "finalbody", []):
                    if isinstance(final, list):
                        _visit_nodes(final, parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                    elif isinstance(final, ast.AST):
                        _visit_nodes([final], parent_fqn=parent_fqn, parent_is_class=parent_is_class)
                for case in getattr(node, "cases", []):
                    if hasattr(case, "body") and isinstance(case.body, list):
                        _visit_nodes(case.body, parent_fqn=parent_fqn, parent_is_class=parent_is_class)

    _visit_nodes(tree.body, parent_fqn=None, parent_is_class=False)
    return ParseResult(symbols=symbols, success=True)


def resolve_enclosing_symbol(
    line: int,
    symbols: list[CodeSymbol],
) -> Optional[CodeSymbol]:
    """Find the narrowest enclosing CodeSymbol for a given 1-based line number.

    Returns:
        The enclosing CodeSymbol with the smallest span (line_end - line_start),
        or None if no symbol encloses the line (mapping to module-level).
    """
    candidates = [
        s for s in symbols
        if s.line_start <= line <= s.line_end
    ]
    if not candidates:
        return None

    # Narrowest span wins. Ties broken by greater start line (innermost).
    return min(candidates, key=lambda s: (s.line_end - s.line_start, -s.line_start))


def analyze_file_diff_symbols(
    file_diff: FileDiff,
    base_source: Optional[str],
    target_source: Optional[str],
) -> FileSymbolChanges:
    """Analyze symbol-level changes for a single FileDiff given raw source revisions.

    Args:
        file_diff:     Parsed FileDiff from diff_analyzer.
        base_source:   Raw source code at base revision, or None if file was added.
        target_source: Raw source code at target revision, or None if file was deleted.

    Returns:
        FileSymbolChanges with classified symbols and parse statuses.
    """
    file_path = file_diff.path
    result = FileSymbolChanges(
        file_path=file_path,
        old_path=file_diff.old_path,
        new_path=file_diff.new_path,
        change_type=file_diff.change_type,
    )

    base_is_py = bool(file_diff.old_path and file_diff.old_path.endswith(".py"))
    target_is_py = bool(file_diff.new_path and file_diff.new_path.endswith(".py"))

    # A file is analyzed as Python only if both revisions (when present) are Python.
    # If a file is renamed from .py to a non-Python extension (e.g. .txt), or vice versa,
    # cross-language AST comparison is unsupported; we fall back to module-level tracking.
    both_paths_present = bool(file_diff.old_path and file_diff.new_path)
    if both_paths_present and (base_is_py != target_is_py):
        is_python = False
    else:
        is_python = target_is_py if file_diff.new_path else base_is_py

    if not is_python:
        # Non-Python or cross-extension file: aggregate all changes under <module> without AST parsing
        all_lines = sorted(set(file_diff.all_added_lines + file_diff.all_deleted_lines))
        if all_lines:
            category = ChangeCategory.MODIFIED
            if file_diff.change_type == FileChangeType.ADDED:
                category = ChangeCategory.ADDED
            elif file_diff.change_type == FileChangeType.DELETED:
                category = ChangeCategory.DELETED

            # Determine line range using actual source content line count
            if target_source is not None:
                line_end = max(1, len(target_source.splitlines()))
                line_start = 1
            elif base_source is not None:
                line_end = max(1, len(base_source.splitlines()))
                line_start = 1
            else:
                # Source unavailable: clearly bounded fallback to the changed lines span
                line_start = min(all_lines)
                line_end = max(all_lines)

            result.changed_symbols.append(
                ChangedSymbol(
                    symbol_name="<module>",
                    fully_qualified_name="<module>",
                    symbol_type=SymbolType.MODULE,
                    change_category=category,
                    line_start=line_start,
                    line_end=line_end,
                    changed_lines=all_lines,
                )
            )
        return result

    # 1. Parse base and target revisions
    base_result: ParseResult
    if base_source is not None:
        base_result = parse_source_symbols(base_source)
        result.base_parse_success = base_result.success
        result.base_parse_error = base_result.error_message
    else:
        base_result = ParseResult(symbols=[], success=True)

    target_result: ParseResult
    if target_source is not None:
        target_result = parse_source_symbols(target_source)
        result.target_parse_success = target_result.success
        result.target_parse_error = target_result.error_message
    else:
        target_result = ParseResult(symbols=[], success=True)

    # 2. Handle AST parsing failure fallback
    # If either side failed to parse, do not perform false symbol-level classifications.
    if not result.base_parse_success or not result.target_parse_success:
        all_lines = sorted(set(file_diff.all_added_lines + file_diff.all_deleted_lines))
        if all_lines:
            category = ChangeCategory.MODIFIED
            if file_diff.change_type == FileChangeType.ADDED:
                category = ChangeCategory.ADDED
            elif file_diff.change_type == FileChangeType.DELETED:
                category = ChangeCategory.DELETED

            target_lines = len(target_source.splitlines()) if target_source else 1
            base_lines = len(base_source.splitlines()) if base_source else 1
            ref_lines = target_lines if target_source is not None else base_lines

            result.changed_symbols.append(
                ChangedSymbol(
                    symbol_name="<module>",
                    fully_qualified_name="<module>",
                    symbol_type=SymbolType.MODULE,
                    change_category=category,
                    line_start=1,
                    line_end=max(1, ref_lines),
                    changed_lines=all_lines,
                )
            )
        return result

    # 3. Map changed lines to enclosing symbols
    # Symbol identity is (fully_qualified_name, symbol_type) to prevent merging
    # distinct symbol types with the same name.
    target_sym_map: dict[tuple[str, SymbolType], CodeSymbol] = {
        (s.fully_qualified_name, s.symbol_type): s for s in target_result.symbols
    }
    added_evidence: dict[tuple[str, SymbolType], list[int]] = {}
    for line in file_diff.all_added_lines:
        enc = resolve_enclosing_symbol(line, target_result.symbols)
        key = (enc.fully_qualified_name, enc.symbol_type) if enc else MODULE_KEY
        added_evidence.setdefault(key, []).append(line)

    # Base deleted lines map against base symbols
    base_sym_map: dict[tuple[str, SymbolType], CodeSymbol] = {
        (s.fully_qualified_name, s.symbol_type): s for s in base_result.symbols
    }
    deleted_evidence: dict[tuple[str, SymbolType], list[int]] = {}
    for line in file_diff.all_deleted_lines:
        enc = resolve_enclosing_symbol(line, base_result.symbols)
        key = (enc.fully_qualified_name, enc.symbol_type) if enc else MODULE_KEY
        deleted_evidence.setdefault(key, []).append(line)

    # 4. Classify symbol changes based on combined evidence
    all_keys = set(added_evidence.keys()) | set(deleted_evidence.keys())
    changed_symbols: list[ChangedSymbol] = []

    for fqn, sym_type in sorted(all_keys, key=lambda k: (k[0], k[1].value)):
        key = (fqn, sym_type)
        if key == MODULE_KEY:
            mod_lines = sorted(set(added_evidence.get(MODULE_KEY, []) + deleted_evidence.get(MODULE_KEY, [])))
            if mod_lines:
                if target_source is not None and file_diff.change_type != FileChangeType.DELETED:
                    cat = ChangeCategory.ADDED if file_diff.change_type == FileChangeType.ADDED else ChangeCategory.MODIFIED
                    total_lines = len(target_source.splitlines())
                else:
                    cat = ChangeCategory.DELETED
                    total_lines = len(base_source.splitlines()) if base_source else 1

                changed_symbols.append(
                    ChangedSymbol(
                        symbol_name="<module>",
                        fully_qualified_name="<module>",
                        symbol_type=SymbolType.MODULE,
                        change_category=cat,
                        line_start=1,
                        line_end=max(1, total_lines),
                        changed_lines=mod_lines,
                    )
                )
            continue

        in_target = key in target_sym_map
        in_base = key in base_sym_map

        if in_target and in_base:
            # Present in both revisions with matching symbol type -> MODIFIED
            t_sym = target_sym_map[key]
            lines = sorted(set(added_evidence.get(key, []) + deleted_evidence.get(key, [])))
            changed_symbols.append(
                ChangedSymbol(
                    symbol_name=t_sym.name,
                    fully_qualified_name=fqn,
                    symbol_type=sym_type,
                    change_category=ChangeCategory.MODIFIED,
                    line_start=t_sym.line_start,  # Target rev per spec
                    line_end=t_sym.line_end,
                    changed_lines=lines,
                )
            )
        elif in_target and not in_base:
            # Present only in target (or changed type) -> ADDED
            t_sym = target_sym_map[key]
            lines = sorted(set(added_evidence.get(key, [])))
            changed_symbols.append(
                ChangedSymbol(
                    symbol_name=t_sym.name,
                    fully_qualified_name=fqn,
                    symbol_type=sym_type,
                    change_category=ChangeCategory.ADDED,
                    line_start=t_sym.line_start,  # Target rev per spec
                    line_end=t_sym.line_end,
                    changed_lines=lines,
                )
            )
        elif in_base and not in_target:
            # Present only in base (or changed type) -> DELETED
            b_sym = base_sym_map[key]
            lines = sorted(set(deleted_evidence.get(key, [])))
            changed_symbols.append(
                ChangedSymbol(
                    symbol_name=b_sym.name,
                    fully_qualified_name=fqn,
                    symbol_type=sym_type,
                    change_category=ChangeCategory.DELETED,
                    line_start=b_sym.line_start,  # Base rev per spec
                    line_end=b_sym.line_end,
                    changed_lines=lines,
                )
            )

    # Sort deterministically by line_start, fully_qualified_name, and symbol_type
    changed_symbols.sort(key=lambda s: (s.line_start, s.fully_qualified_name, s.symbol_type.value))
    result.changed_symbols = changed_symbols
    return result


def analyze_repository_symbols(
    repository_path: str,
    base_revision: str,
    target_revision: str,
    parsed_diff: ParsedDiff,
) -> list[FileSymbolChanges]:
    """Analyze symbol changes across all files in a repository between two revisions.

    Args:
        repository_path: Path to the Git repository.
        base_revision:   Git revision for the base state.
        target_revision: Git revision for the target state.
        parsed_diff:     ParsedDiff instance containing FileDiffs.

    Returns:
        List of FileSymbolChanges, one per file in the parsed diff.

    Raises:
        GitServiceError: If the repository or revision specifier is invalid.
    """
    repo_path = validate_repository(repository_path)
    validate_revision(repo_path, base_revision)
    validate_revision(repo_path, target_revision)

    results: list[FileSymbolChanges] = []

    for file_diff in parsed_diff.files:
        base_source: Optional[str] = None
        target_source: Optional[str] = None

        # Retrieve base revision content if file existed in base
        if file_diff.old_path and file_diff.change_type != FileChangeType.ADDED:
            base_source = get_file_content_at_revision(
                repository_path, base_revision, file_diff.old_path
            )

        # Retrieve target revision content if file exists in target
        if file_diff.new_path and file_diff.change_type != FileChangeType.DELETED:
            target_source = get_file_content_at_revision(
                repository_path, target_revision, file_diff.new_path
            )

        file_changes = analyze_file_diff_symbols(file_diff, base_source, target_source)
        results.append(file_changes)

    return results
