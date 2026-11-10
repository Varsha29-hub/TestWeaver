"""Test Impact Analysis service for TestWeaver.

Week 3, Phase 3:
Stage 1: Identifies potentially affected tests by correlating changed source files
         from AST symbol changes with imported modules extracted during test discovery.
Stage 2: Integrates fine-grained symbol dependencies from Python test ASTs, resolving
         symbols, handling parameter shadowing, exact multi-scope AST verification,
         relative import resolution via importlib.util.resolve_name, and strict
         uncertainty fallback.

Architectural boundaries and design rules:
1. Canonical Module Paths:
   Source file paths are converted to canonical dot-separated module paths
   (e.g. 'src/calculator.py' -> 'src.calculator', 'lib/helpers.py' -> 'lib.helpers').
   Source root prefixes such as 'src.' or 'lib.' are NOT heuristically stripped
   unless explicit repository configuration establishes them as import roots.
2. Relative Imports:
   Relative imports are resolved using Python's PEP 328 standard semantics via
   importlib.util.resolve_name when the test package anchor is known. Syntactically
   invalid imports (ValueError) are excluded from internal matching. Unanchored
   relative imports fall back conservatively if repo changes exist.
3. Substring Protection:
   Matches must be exact or separated by a dot delimiter to prevent collisions
   between similar names (e.g. 'src.calculator' never matches 'src.calculator_extra').
4. Multi-Scope AST Verification:
   Traverses nested function, class, and method AST chains to confirm whether a
   referenced symbol was ADDED, DELETED, MODIFIED, or UNCHANGED in revision source ASTs.
5. Strict Uncertainty Handling:
   Unresolved dynamic reflection, dynamic imports, wildcard imports, and unknown
   factory calls trigger conservative fallback rather than false UNAFFECTED.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

from services.ast_analyzer import (
    ChangeCategory,
    ChangedSymbol,
    FileSymbolChanges,
    SymbolType,
    get_node_end_line,
    get_node_start_line,
)
from services.diff_analyzer import FileChangeType
from services.git_service import GitServiceError, get_file_content_at_revision
from services.test_discovery import DiscoveredTest, DiscoveryResult
from services.test_symbol_extractor import (
    TestSymbolDependencies,
    TestSymbolExtractor,
)

logger = logging.getLogger(__name__)


class ImpactType(str, Enum):
    """Classification of test impact granularity."""

    MODULE = "MODULE"
    SYMBOL = "SYMBOL"


class ImpactConfidence(str, Enum):
    """Confidence level of test impact assessment."""

    CONFIDENT = "CONFIDENT"
    CONSERVATIVE = "CONSERVATIVE"


class VerificationStatus(str, Enum):
    """Status of symbol verification in revision AST."""

    FOUND = "FOUND"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SYNTAX_ERROR = "SYNTAX_ERROR"


@dataclass
class TestImpact:
    """Represents a discovered test affected by source code changes."""

    # 7 Stage 1 positional fields preserved in exact order:
    node_id: str
    test_file: str
    test_name: str
    impact_type: ImpactType
    matched_file: str
    matched_module: str
    reasons: list[str] = field(default_factory=list)
    # Stage 2 optional fields with defaults:
    matched_symbol: Optional[str] = None
    confidence: Optional[ImpactConfidence] = None

    def __iter__(self):
        """Enable unpacking of the 7 Stage 1 positional fields for backward compatibility."""
        return iter((
            self.node_id,
            self.test_file,
            self.test_name,
            self.impact_type,
            self.matched_file,
            self.matched_module,
            self.reasons,
        ))


@dataclass
class ImpactAnalysisResult:
    """Aggregated results of the test impact analysis."""

    total_discovered_tests: int
    total_affected_tests: int
    affected_tests: list[TestImpact] = field(default_factory=list)
    unaffected_tests: list[str] = field(default_factory=list)


def file_to_module_path(file_path: str) -> Optional[str]:
    """Convert a Python source file path to a canonical Python module path.

    Handles POSIX and Windows path separators.
    Maps package __init__.py files to their enclosing package module name.
    Returns None for non-Python files or invalid paths.

    Examples:
        'src/calculator.py' -> 'src.calculator'
        'src\\\\utils\\\\string_helpers.py' -> 'src.utils.string_helpers'
        'calculator.py' -> 'calculator'
        'lib/helpers.py' -> 'lib.helpers'
        'src/utils/__init__.py' -> 'src.utils'
        'README.md' -> None
    """
    if not file_path:
        return None

    # Normalize Windows backslashes to forward slashes and strip whitespace
    normalized = file_path.strip().replace("\\", "/")

    # Ensure actual file extension is .py
    if not normalized.endswith(".py"):
        return None

    # Strip .py extension
    without_ext = normalized[:-3]

    # Split into path components, filtering out empty segments and '.'
    parts = [p for p in without_ext.split("/") if p and p != "."]
    if not parts:
        return None

    # Handle package __init__.py -> maps to parent package
    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        # e.g. root __init__.py without enclosing package
        return None

    return ".".join(parts)


def get_importable_module_paths(file_path: str) -> list[str]:
    """Return the canonical importable module path for a source file.

    Source-root stripping (e.g. discarding 'src.' or 'lib.') is disallowed
    without explicit repository configuration. Only canonical paths are used.
    """
    canonical = file_to_module_path(file_path)
    if not canonical:
        return []
    return [canonical]


def is_module_affected(changed_module: str, imported_module: str) -> bool:
    """Check if an imported module or symbol matches or is affected by a changed module.

    Supports:
      1. Exact module match:
         imported_module == changed_module (e.g. 'src.calculator' == 'src.calculator')
      2. Imported symbol or submodule of the changed module:
         imported_module starts with changed_module + '.' (e.g. 'src.calculator.add')
      3. Changed package with imported submodule:
         e.g. changed 'src.utils', imported 'src.utils.string_helpers'

    Rejects:
      - Accidental substring matches: 'src.calculator' does NOT match 'src.calculator_extra'.
      - Relative imports: Relative imports with leading dots (e.g. '.calculator', '..utils')
        are not naively dot-stripped and remain unmatched in Stage 1 because reliable
        package anchors are not available from test discovery data alone.
    """
    if not changed_module or not imported_module:
        return False

    # Relative imports cannot be safely resolved without full package context;
    # reject naive dot-stripping in Stage 1.
    if imported_module.startswith("."):
        return False

    # 1. Exact match
    if imported_module == changed_module:
        return True

    # 2. Submodule or symbol import: must be strictly followed by a dot
    if imported_module.startswith(changed_module + "."):
        return True

    return False


def extract_changed_module_mappings(
    symbol_changes: list[FileSymbolChanges],
) -> list[tuple[str, str, str]]:
    """Extract (file_path, module_name, context_reason) tuples from symbol changes.

    Handles additions, deletions, modifications, and renames (both old and new paths).
    Ignores non-Python files.
    """
    mappings: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for change in symbol_changes:
        if change.change_type == FileChangeType.RENAMED:
            # Old path in base revision
            if change.old_path:
                for mod in get_importable_module_paths(change.old_path):
                    pair = (change.old_path, mod)
                    if pair not in seen:
                        seen.add(pair)
                        mappings.append((change.old_path, mod, f"renamed from '{change.old_path}'"))

            # New path in target revision
            if change.new_path:
                for mod in get_importable_module_paths(change.new_path):
                    pair = (change.new_path, mod)
                    if pair not in seen:
                        seen.add(pair)
                        mappings.append((change.new_path, mod, f"renamed to '{change.new_path}'"))

        elif change.change_type == FileChangeType.DELETED:
            path = change.old_path or change.file_path
            for mod in get_importable_module_paths(path):
                pair = (path, mod)
                if pair not in seen:
                    seen.add(pair)
                    mappings.append((path, mod, f"deleted file '{path}'"))

        elif change.change_type == FileChangeType.ADDED:
            path = change.new_path or change.file_path
            for mod in get_importable_module_paths(path):
                pair = (path, mod)
                if pair not in seen:
                    seen.add(pair)
                    mappings.append((path, mod, f"added file '{path}'"))

        else:  # MODIFIED or default
            path = change.new_path or change.file_path
            for mod in get_importable_module_paths(path):
                pair = (path, mod)
                if pair not in seen:
                    seen.add(pair)
                    mappings.append((path, mod, f"modified file '{path}'"))

            if change.old_path and change.old_path != path:
                for mod in get_importable_module_paths(change.old_path):
                    pair = (change.old_path, mod)
                    if pair not in seen:
                        seen.add(pair)
                        mappings.append((change.old_path, mod, f"modified file '{change.old_path}'"))

    return mappings


def _find_scope_in_stmts(stmts: list[ast.AST], name: str) -> Optional[ast.AST]:
    """Find an enclosing scope AST node (ClassDef, FunctionDef, AsyncFunctionDef) by name."""
    for stmt in stmts:
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == name:
                return stmt
        if isinstance(stmt, ast.If):
            found = _find_scope_in_stmts(stmt.body, name) or _find_scope_in_stmts(stmt.orelse, name)
            if found:
                return found
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            found = _find_scope_in_stmts(stmt.body, name) or _find_scope_in_stmts(stmt.orelse, name)
            if found:
                return found
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            found = _find_scope_in_stmts(stmt.body, name)
            if found:
                return found
        elif isinstance(stmt, ast.Try):
            found = _find_scope_in_stmts(stmt.body, name)
            if found:
                return found
            for handler in stmt.handlers:
                found = _find_scope_in_stmts(handler.body, name)
                if found:
                    return found
            found = _find_scope_in_stmts(stmt.orelse, name) or _find_scope_in_stmts(stmt.finalbody, name)
            if found:
                return found
        elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            for case in stmt.cases:
                found = _find_scope_in_stmts(case.body, name)
                if found:
                    return found
    return None


def _target_contains_name(target_node: ast.AST, name: str) -> bool:
    """Check if an assignment target (including unpacked tuples/lists and starred targets) contains name."""
    if isinstance(target_node, ast.Name):
        return target_node.id == name
    if isinstance(target_node, ast.Starred):
        return _target_contains_name(target_node.value, name)
    if isinstance(target_node, (ast.Tuple, ast.List)):
        return any(_target_contains_name(elt, name) for elt in target_node.elts)
    return False


def _extract_module_imports(
    source_code: Optional[str],
    module_name: Optional[str] = None,
    is_package: bool = False,
) -> set[str]:
    """Extract all imported module names from source code, resolving relative imports if module_name is provided."""
    if not source_code:
        return set()
    try:
        tree = ast.parse(source_code)
    except Exception:
        return set()

    pkg_anchor: Optional[str] = None
    if module_name:
        if is_package:
            pkg_anchor = module_name
        elif "." in module_name:
            pkg_anchor = module_name.rsplit(".", 1)[0]
        else:
            pkg_anchor = ""

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if pkg_anchor is not None:
                    if node.module:
                        rel_name = f"{'.' * node.level}{node.module}"
                        try:
                            resolved = importlib.util.resolve_name(rel_name, pkg_anchor)
                            imported.add(resolved)
                        except (ValueError, ImportError):
                            pass
                    else:
                        for a in node.names:
                            if a.name != "*":
                                rel_name = f"{'.' * node.level}{a.name}"
                                try:
                                    resolved = importlib.util.resolve_name(rel_name, pkg_anchor)
                                    imported.add(resolved)
                                except (ValueError, ImportError):
                                    pass
            else:
                if node.module:
                    imported.add(node.module)
    return imported


def _find_symbol_in_scope_stmts(stmts: list[ast.AST], name: str) -> Optional[ast.AST]:
    """Find a target symbol node (function, method, class, or assignment) by name."""
    for stmt in stmts:
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == name:
                return stmt
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if _target_contains_name(target, name):
                    return stmt
        elif isinstance(stmt, ast.AnnAssign):
            if _target_contains_name(stmt.target, name):
                return stmt
        if isinstance(stmt, ast.If):
            found = _find_symbol_in_scope_stmts(stmt.body, name) or _find_symbol_in_scope_stmts(stmt.orelse, name)
            if found:
                return found
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            found = _find_symbol_in_scope_stmts(stmt.body, name) or _find_symbol_in_scope_stmts(stmt.orelse, name)
            if found:
                return found
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            found = _find_symbol_in_scope_stmts(stmt.body, name)
            if found:
                return found
        elif isinstance(stmt, ast.Try):
            found = _find_symbol_in_scope_stmts(stmt.body, name)
            if found:
                return found
            for handler in stmt.handlers:
                found = _find_symbol_in_scope_stmts(handler.body, name)
                if found:
                    return found
            found = _find_symbol_in_scope_stmts(stmt.orelse, name) or _find_symbol_in_scope_stmts(stmt.finalbody, name)
            if found:
                return found
        elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            for case in stmt.cases:
                found = _find_symbol_in_scope_stmts(case.body, name)
                if found:
                    return found
    return None


def verify_symbol_in_source(
    source_code_or_tree: Union[Optional[str], ast.AST],
    qualified_name: str,
) -> tuple[VerificationStatus, Optional[ast.AST]]:
    """Traverse AST to verify presence of a symbol across arbitrary nested scopes.

    Accepts either raw source code string (which will be parsed) or a pre-parsed ast.AST.

    Returns:
        (VerificationStatus, matched_ast_node_or_None)
    """
    if source_code_or_tree is None:
        return (VerificationStatus.SOURCE_UNAVAILABLE, None)

    if isinstance(source_code_or_tree, ast.AST):
        tree = source_code_or_tree
    else:
        try:
            tree = ast.parse(source_code_or_tree)
        except SyntaxError:
            return (VerificationStatus.SYNTAX_ERROR, None)
        except Exception:
            return (VerificationStatus.SYNTAX_ERROR, None)

    segments = [s for s in qualified_name.split(".") if s]
    if not segments:
        return (VerificationStatus.CONFIRMED_ABSENT, None)

    current_stmts = tree.body
    for seg in segments[:-1]:
        scope_node = _find_scope_in_stmts(current_stmts, seg)
        if scope_node is None or not hasattr(scope_node, "body"):
            return (VerificationStatus.SCOPE_NOT_FOUND, None)
        current_stmts = scope_node.body

    target_name = segments[-1]
    matched_node = _find_symbol_in_scope_stmts(current_stmts, target_name)
    if matched_node is not None:
        return (VerificationStatus.FOUND, matched_node)

    return (VerificationStatus.CONFIRMED_ABSENT, None)


def analyze_test_impact(
    symbol_changes: list[FileSymbolChanges],
    discovery_result: DiscoveryResult,
    *,
    repository_path: Optional[str] = None,
    base_revision: Optional[str] = None,
    target_revision: Optional[str] = None,
    source_provider: Optional[Callable[[str, str], Optional[str]]] = None,
    test_source_provider: Optional[Callable[[str], Optional[str]]] = None,
) -> ImpactAnalysisResult:
    """Analyze test impact using AST symbol dependency extraction and multi-scope verification.

    Integrates Stage 1 module correlation with Stage 2 fine-grained symbol references.
    Falls back gracefully to Stage 1 module correlation when test source ASTs or symbol
    evidence are not available, ensuring full backward compatibility.

    Args:
        symbol_changes:       List of FileSymbolChanges from diff/AST analysis.
        discovery_result:     DiscoveryResult containing discovered tests.
        repository_path:      Optional path to git repository for revision content loading.
        base_revision:        Optional base Git revision hash/ref.
        target_revision:      Optional target Git revision hash/ref.
        source_provider:      Optional callback(rev_label, file_path) -> Optional[str].
        test_source_provider: Optional callback(file_path) -> Optional[str].

    Returns:
        ImpactAnalysisResult with affected tests, granular reasons, and unaffected tests.
    """
    total_discovered = discovery_result.total_tests

    # Extract all candidate changed modules from symbol changes
    changed_modules = sorted(
        extract_changed_module_mappings(symbol_changes),
        key=lambda m: (m[0], m[1], m[2]),
    )

    # Detect modules containing the '<module>' sentinel change
    modules_with_sentinel: set[str] = set()
    for change in symbol_changes:
        for cs in change.changed_symbols:
            if cs.fully_qualified_name == "<module>" or cs.symbol_name == "<module>":
                for p in (change.new_path, change.file_path, change.old_path):
                    if p:
                        for m in get_importable_module_paths(p):
                            modules_with_sentinel.add(m)

    # Index changed symbols by their full project-level FQN: 'module.symbol_fqn'
    changed_symbols_by_fqn: dict[str, list[tuple[FileSymbolChanges, ChangedSymbol, str]]] = defaultdict(list)
    for change in symbol_changes:
        candidate_paths: list[str] = []
        if change.new_path:
            candidate_paths.append(change.new_path)
        if change.file_path and change.file_path not in candidate_paths:
            candidate_paths.append(change.file_path)
        if change.old_path and change.old_path not in candidate_paths:
            candidate_paths.append(change.old_path)

        for p in candidate_paths:
            for mod in get_importable_module_paths(p):
                for cs in change.changed_symbols:
                    if cs.fully_qualified_name == "<module>":
                        continue
                    full_fqn = f"{mod}.{cs.fully_qualified_name}"
                    changed_symbols_by_fqn[full_fqn].append((change, cs, p))

    # Source and AST caches to avoid repeated Git subprocess calls and AST parsing
    source_cache: dict[tuple[str, str], Optional[str]] = {}
    ast_cache: dict[tuple[str, str], tuple[VerificationStatus, Optional[ast.AST]]] = {}

    def get_source(rev_label: str, file_path: str) -> Optional[str]:
        cache_key = (rev_label, file_path)
        if cache_key in source_cache:
            return source_cache[cache_key]

        content: Optional[str] = None
        if source_provider is not None:
            content = source_provider(rev_label, file_path)
        elif repository_path is not None:
            rev = base_revision if rev_label == "base" else target_revision
            if rev:
                try:
                    content = get_file_content_at_revision(repository_path, rev, file_path)
                except Exception:
                    content = None

        source_cache[cache_key] = content
        return content

    def get_revision_ast(rev_label: str, file_path: str) -> tuple[VerificationStatus, Optional[ast.AST]]:
        cache_key = (rev_label, file_path)
        if cache_key in ast_cache:
            return ast_cache[cache_key]

        src = get_source(rev_label, file_path)
        if src is None:
            res = (VerificationStatus.SOURCE_UNAVAILABLE, None)
        else:
            try:
                tree = ast.parse(src)
                res = (VerificationStatus.FOUND, tree)
            except SyntaxError:
                res = (VerificationStatus.SYNTAX_ERROR, None)
            except Exception:
                res = (VerificationStatus.SYNTAX_ERROR, None)

        ast_cache[cache_key] = res
        return res

    def get_test_source(file_path: str) -> Optional[str]:
        if test_source_provider is not None:
            return test_source_provider(file_path)
        candidate_paths: list[Path] = []
        if repository_path:
            candidate_paths.append(Path(repository_path) / file_path)
        candidate_paths.append(Path(file_path))
        for p in candidate_paths:
            try:
                if p.is_file():
                    return p.read_text(encoding="utf-8")
            except Exception:
                pass
        return None

    # Cache for test file parsed AST / extraction results
    test_file_deps_cache: dict[str, Union[list[TestSymbolDependencies], str]] = {}

    def get_extracted_deps(file_path: str) -> Union[list[TestSymbolDependencies], str]:
        if file_path in test_file_deps_cache:
            return test_file_deps_cache[file_path]
        content = get_test_source(file_path)
        if content is None:
            test_file_deps_cache[file_path] = "UNAVAILABLE"
            return "UNAVAILABLE"
        try:
            ast.parse(content, filename=file_path)
        except SyntaxError:
            test_file_deps_cache[file_path] = "SYNTAX_ERROR"
            return "SYNTAX_ERROR"
        except Exception:
            test_file_deps_cache[file_path] = "SYNTAX_ERROR"
            return "SYNTAX_ERROR"

        extractor = TestSymbolExtractor(content, file_path)
        deps = extractor.extract()
        test_file_deps_cache[file_path] = deps
        return deps

    affected_tests: list[TestImpact] = []
    unaffected_tests: list[str] = []

    for test in discovery_result.tests:
        norm_test_file = test.file_path.strip().replace("\\", "/")
        deps_or_status = get_extracted_deps(test.file_path)

        # Case 1: Test file has SyntaxError
        if deps_or_status == "SYNTAX_ERROR":
            if changed_modules:
                primary_file, primary_mod, _ = changed_modules[0]
                affected_tests.append(
                    TestImpact(
                        node_id=test.node_id,
                        test_file=test.file_path,
                        test_name=test.test_name,
                        impact_type=ImpactType.MODULE,
                        matched_file=primary_file,
                        matched_module=primary_mod,
                        reasons=[
                            f"Test file '{test.file_path}' contains syntax errors while repository has Python changes"
                        ],
                        confidence=ImpactConfidence.CONSERVATIVE,
                    )
                )
            else:
                unaffected_tests.append(test.node_id)
            continue

        # Case 2: Test file source is unavailable (e.g. Stage 1 mock tests)
        if deps_or_status == "UNAVAILABLE":
            test_reasons: list[str] = []
            primary_matched_file: str = ""
            primary_matched_module: str = ""
            matched_pairs: set[tuple[str, str]] = set()

            for file_path, changed_mod, context in changed_modules:
                for imp in test.imported_modules:
                    if is_module_affected(changed_mod, imp):
                        pair = (file_path, changed_mod)
                        if pair not in matched_pairs:
                            matched_pairs.add(pair)
                            if not primary_matched_file:
                                primary_matched_file = file_path
                                primary_matched_module = changed_mod
                            if imp == changed_mod:
                                reason = f"Test imports module '{changed_mod}' directly ({context})"
                            elif imp.startswith(changed_mod + "."):
                                reason = f"Test imports '{imp}' from changed module '{changed_mod}' ({context})"
                            else:
                                reason = f"Test imports affected module '{imp}' ({context})"
                            test_reasons.append(reason)

            if test_reasons:
                affected_tests.append(
                    TestImpact(
                        node_id=test.node_id,
                        test_file=test.file_path,
                        test_name=test.test_name,
                        impact_type=ImpactType.MODULE,
                        matched_file=primary_matched_file,
                        matched_module=primary_matched_module,
                        reasons=test_reasons,
                    )
                )
            else:
                unaffected_tests.append(test.node_id)
            continue

        # Case 3: Test file parsed into TestSymbolDependencies list
        dep: Optional[TestSymbolDependencies] = None
        for d in deps_or_status:
            if d.node_id == test.node_id or d.test_name == test.test_name:
                dep = d
                break
            if d.test_file == norm_test_file and d.test_name == test.test_name:
                dep = d
                break

        if dep is None:
            # Fall back to Stage 1 module-level matching if extractor didn't map test
            test_reasons = []
            primary_matched_file = ""
            primary_matched_module = ""
            matched_pairs = set()

            for file_path, changed_mod, context in changed_modules:
                for imp in test.imported_modules:
                    if is_module_affected(changed_mod, imp):
                        pair = (file_path, changed_mod)
                        if pair not in matched_pairs:
                            matched_pairs.add(pair)
                            if not primary_matched_file:
                                primary_matched_file = file_path
                                primary_matched_module = changed_mod
                            test_reasons.append(f"Test imports module '{changed_mod}' ({context})")

            if test_reasons:
                affected_tests.append(
                    TestImpact(
                        node_id=test.node_id,
                        test_file=test.file_path,
                        test_name=test.test_name,
                        impact_type=ImpactType.MODULE,
                        matched_file=primary_matched_file,
                        matched_module=primary_matched_module,
                        reasons=test_reasons,
                    )
                )
            else:
                unaffected_tests.append(test.node_id)
            continue

        # Step 3A: Check Relative Imports via PEP 328 resolution
        if dep.unresolved_relative_imports:
            test_mod = file_to_module_path(norm_test_file)
            if norm_test_file.endswith("__init__.py"):
                pkg_anchor = test_mod
            else:
                pkg_anchor = test_mod.rsplit(".", 1)[0] if (test_mod and "." in test_mod) else None

            if pkg_anchor is None:
                # Unanchored relative import
                if changed_modules:
                    primary_file, primary_mod, _ = changed_modules[0]
                    affected_tests.append(
                        TestImpact(
                            node_id=test.node_id,
                            test_file=test.file_path,
                            test_name=test.test_name,
                            impact_type=ImpactType.MODULE,
                            matched_file=primary_file,
                            matched_module=primary_mod,
                            reasons=[
                                f"Test file '{test.file_path}' contains relative import '{dep.unresolved_relative_imports[0]}' outside package context"
                            ],
                            confidence=ImpactConfidence.CONSERVATIVE,
                        )
                    )
                    continue
            else:
                rel_matched = False
                rel_reasons: list[str] = []
                primary_rel_file = ""
                primary_rel_mod = ""
                for rel_imp in dep.unresolved_relative_imports:
                    try:
                        resolved_name = importlib.util.resolve_name(rel_imp, pkg_anchor)
                    except (ValueError, ImportError):
                        # Syntactically invalid traversal above top-level package
                        continue
                    for chg_file, chg_mod, context in changed_modules:
                        if is_module_affected(chg_mod, resolved_name):
                            rel_matched = True
                            if not primary_rel_file:
                                primary_rel_file = chg_file
                                primary_rel_mod = chg_mod
                            rel_reasons.append(
                                f"Test relative import '{rel_imp}' resolves to affected module '{resolved_name}' ({context})"
                            )
                if rel_matched:
                    affected_tests.append(
                        TestImpact(
                            node_id=test.node_id,
                            test_file=test.file_path,
                            test_name=test.test_name,
                            impact_type=ImpactType.MODULE,
                            matched_file=primary_rel_file,
                            matched_module=primary_rel_mod,
                            reasons=sorted(rel_reasons),
                            confidence=ImpactConfidence.CONSERVATIVE,
                        )
                    )
                    continue

        # Step 3B: Check Strict Uncertainty (Dynamic reflection, unknown factory instances, wildcard imports)
        remaining_uncertainties = [
            r for r in dep.uncertainty_reasons
            if not r.startswith("Relative import 'from ")
        ]
        if dep.has_uncertainty and remaining_uncertainties:
            if changed_modules:
                primary_file, primary_mod, _ = changed_modules[0]
                affected_tests.append(
                    TestImpact(
                        node_id=test.node_id,
                        test_file=test.file_path,
                        test_name=test.test_name,
                        impact_type=ImpactType.MODULE,
                        matched_file=primary_file,
                        matched_module=primary_mod,
                        reasons=[
                            f"Test contains unresolved dynamic uncertainty that cannot be proven disjoint from changed code: {r}"
                            for r in remaining_uncertainties
                        ],
                        confidence=ImpactConfidence.CONSERVATIVE,
                    )
                )
                continue

        # Step 3C: Check Module-Level Sentinel change ('<module>' symbol change in imported module)
        module_sentinel_matched = False
        module_sentinel_reasons: list[str] = []
        primary_sentinel_file = ""
        primary_sentinel_mod = ""
        all_test_imports = sorted(set(dep.imported_modules) | set(test.imported_modules))

        for chg_file, chg_mod, context in changed_modules:
            if chg_mod in modules_with_sentinel:
                for imp in all_test_imports:
                    if is_module_affected(chg_mod, imp):
                        module_sentinel_matched = True
                        if not primary_sentinel_file:
                            primary_sentinel_file = chg_file
                            primary_sentinel_mod = chg_mod
                        module_sentinel_reasons.append(
                            f"Test imports module '{chg_mod}' with module-level modifications ({context})"
                        )
        if module_sentinel_matched:
            affected_tests.append(
                TestImpact(
                    node_id=test.node_id,
                    test_file=test.file_path,
                    test_name=test.test_name,
                    impact_type=ImpactType.MODULE,
                    matched_file=primary_sentinel_file,
                    matched_module=primary_sentinel_mod,
                    reasons=sorted(module_sentinel_reasons),
                    confidence=ImpactConfidence.CONSERVATIVE,
                )
            )
            continue

        # Step 3D: Symbol-Level Candidate Verification
        candidate_fqns: list[str] = sorted(dep.referenced_symbols)
        for mod in sorted(dep.module_attribute_references.keys()):
            for attr in sorted(dep.module_attribute_references[mod]):
                attr_fqn = f"{mod}.{attr}"
                if attr_fqn not in candidate_fqns:
                    candidate_fqns.append(attr_fqn)

        symbol_matches: list[tuple[str, str, str, str, ImpactConfidence]] = []
        ast_fallback_records: list[tuple[str, str, str]] = []  # (fpath, mod, reason)

        for cand in candidate_fqns:
            if cand in changed_symbols_by_fqn:
                for change, cs, fpath in changed_symbols_by_fqn[cand]:
                    mod = cand[: -(len(cs.fully_qualified_name) + 1)]
                    base_rev_path = change.old_path or change.file_path
                    target_rev_path = change.new_path or change.file_path

                    if source_provider is not None or repository_path is not None:
                        # Legitimate file addition: file legitimately does not exist in base revision
                        if change.change_type == FileChangeType.ADDED or change.old_path is None:
                            base_status = VerificationStatus.CONFIRMED_ABSENT
                            base_node = None
                        else:
                            b_status, b_tree = get_revision_ast("base", base_rev_path)
                            if b_status == VerificationStatus.FOUND and b_tree is not None:
                                base_status, base_node = verify_symbol_in_source(b_tree, cs.fully_qualified_name)
                            else:
                                base_status = b_status
                                base_node = None

                        # Legitimate file deletion: file legitimately does not exist in target revision
                        if change.change_type == FileChangeType.DELETED or change.new_path is None:
                            target_status = VerificationStatus.CONFIRMED_ABSENT
                            target_node = None
                        else:
                            t_status, t_tree = get_revision_ast("target", target_rev_path)
                            if t_status == VerificationStatus.FOUND and t_tree is not None:
                                target_status, target_node = verify_symbol_in_source(t_tree, cs.fully_qualified_name)
                            else:
                                target_status = t_status
                                target_node = None

                        if target_status in (VerificationStatus.SOURCE_UNAVAILABLE, VerificationStatus.SYNTAX_ERROR) or \
                           base_status in (VerificationStatus.SOURCE_UNAVAILABLE, VerificationStatus.SYNTAX_ERROR):
                            ast_fallback_records.append(
                                (fpath, mod, f"Missing revision AST or syntax error verifying '{cand}'")
                            )
                            continue

                        if cs.change_category == ChangeCategory.ADDED:
                            if target_status == VerificationStatus.FOUND and base_status in (
                                VerificationStatus.CONFIRMED_ABSENT,
                                VerificationStatus.SCOPE_NOT_FOUND,
                            ):
                                reason = f"Test references added symbol '{cand}' in '{fpath}'"
                                symbol_matches.append((cand, fpath, mod, reason, ImpactConfidence.CONFIDENT))
                        elif cs.change_category == ChangeCategory.DELETED:
                            if base_status == VerificationStatus.FOUND and target_status in (
                                VerificationStatus.CONFIRMED_ABSENT,
                                VerificationStatus.SCOPE_NOT_FOUND,
                            ):
                                reason = f"Test references deleted symbol '{cand}' in '{fpath}'"
                                symbol_matches.append((cand, fpath, mod, reason, ImpactConfidence.CONFIDENT))
                        elif cs.change_category == ChangeCategory.MODIFIED:
                            if target_status == VerificationStatus.FOUND and base_status == VerificationStatus.FOUND:
                                if cs.changed_lines:
                                    reason = f"Test references modified symbol '{cand}' in '{fpath}'"
                                    symbol_matches.append((cand, fpath, mod, reason, ImpactConfidence.CONFIDENT))
                            else:
                                # Structural mismatch: symbol marked MODIFIED cannot be symmetrically verified in both revisions
                                reason = (
                                    f"Symbol '{cand}' marked MODIFIED could not be symmetrically verified in both revisions "
                                    f"(target={target_status.value}, base={base_status.value})"
                                )
                                ast_fallback_records.append((fpath, mod, reason))
                    else:
                        # In absence of revision sources, trust ChangedSymbol categorization
                        cat_str = cs.change_category.value.lower()
                        reason = f"Test references {cat_str} symbol '{cand}' in '{fpath}'"
                        symbol_matches.append((cand, fpath, mod, reason, ImpactConfidence.CONFIDENT))

        # Check unverified attribute references on changed modules & cross-file inheritance
        unverified_fallback_candidates: list[tuple[str, str]] = []  # (chg_file, chg_mod)
        unverified_reasons: list[str] = []

        if not symbol_matches:
            for mod in sorted(dep.module_attribute_references.keys()):
                matching_changes = [m for m in changed_modules if m[1] == mod]
                if matching_changes:
                    chg_file, chg_mod, context = matching_changes[0]
                    t_status, t_tree = get_revision_ast("target", chg_file)
                    for attr in sorted(dep.module_attribute_references[mod]):
                        cand_fqn = f"{mod}.{attr}"
                        if cand_fqn in changed_symbols_by_fqn:
                            continue
                        if t_status == VerificationStatus.FOUND and t_tree is not None:
                            status, _ = verify_symbol_in_source(t_tree, attr)
                        else:
                            status = t_status

                        if status == VerificationStatus.FOUND:
                            # Confirmed present in target revision AST and not in changed_symbols (unchanged)
                            continue
                        elif status in (VerificationStatus.CONFIRMED_ABSENT, VerificationStatus.SCOPE_NOT_FOUND):
                            # The attribute does not exist in target class/module AST (dynamic, inherited, or absent)
                            unverified_fallback_candidates.append((chg_file, chg_mod))
                            unverified_reasons.append(
                                f"Test accesses unverified attribute '{attr}' on changed module '{mod}' that cannot be verified in AST"
                            )
                        elif status in (VerificationStatus.SOURCE_UNAVAILABLE, VerificationStatus.SYNTAX_ERROR):
                            unverified_fallback_candidates.append((chg_file, chg_mod))
                            unverified_reasons.append(
                                f"Missing revision AST or syntax error verifying unverified attribute '{attr}' on '{mod}'"
                            )
                else:
                    # mod itself did not change. Check cross-file inheritance:
                    # If an attribute on mod cannot be verified in mod's AST, and mod imports
                    # a module that DID change, flag conservative module impact.
                    candidate_file_paths = [
                        (mod.replace(".", "/") + ".py", False),
                        (mod.replace(".", "/") + "/__init__.py", True),
                    ]
                    mod_src = None
                    is_pkg = False
                    for cfp, is_p in candidate_file_paths:
                        mod_src = get_source("target", cfp) or get_source("base", cfp)
                        if mod_src is not None:
                            is_pkg = is_p
                            break

                    if mod_src is not None:
                        mod_imports = _extract_module_imports(mod_src, mod, is_package=is_pkg)
                        for attr in sorted(dep.module_attribute_references[mod]):
                            status, _ = verify_symbol_in_source(mod_src, attr)
                            if status in (VerificationStatus.CONFIRMED_ABSENT, VerificationStatus.SCOPE_NOT_FOUND):
                                for chg_file, chg_mod, context in changed_modules:
                                    if any(is_module_affected(chg_mod, imp) for imp in mod_imports):
                                        unverified_fallback_candidates.append((chg_file, chg_mod))
                                        unverified_reasons.append(
                                            f"Test calls attribute '{attr}' on module '{mod}' which imports changed module '{chg_mod}' (cross-file inheritance or dependency)"
                                        )
                    else:
                        # Finding 4: mod_src could not be loaded.
                        # Check relevance before flagging conservative fallback:
                        mod_root = mod.split(".")[0]
                        changed_roots = {chg_mod.split(".")[0] for _, chg_mod, _ in changed_modules}
                        test_imports = dep.imported_modules | set(test.imported_modules)
                        imports_changed = any(
                            is_module_affected(chg_mod, imp)
                            for _, chg_mod, _ in changed_modules
                            for imp in test_imports
                        )

                        if mod_root in changed_roots or imports_changed:
                            # Relevant project module whose source is unavailable.
                            # Select genuinely related changed module:
                            # 1. Prefer changed modules sharing the same top-level package
                            related_changes = [m for m in changed_modules if m[1].split(".")[0] == mod_root]
                            # 2. Or changed modules directly imported by this test
                            if not related_changes:
                                related_changes = [
                                    m for m in changed_modules
                                    if any(is_module_affected(m[1], imp) for imp in test_imports)
                                ]
                            if not related_changes:
                                related_changes = changed_modules

                            primary_chg_file, primary_chg_mod, _ = sorted(
                                related_changes, key=lambda m: (m[0], m[1], m[2])
                            )[0]
                            for attr in sorted(dep.module_attribute_references[mod]):
                                unverified_fallback_candidates.append((primary_chg_file, primary_chg_mod))
                                unverified_reasons.append(
                                    f"Test calls unverified attribute '{attr}' on module '{mod}' whose source is unavailable while related module '{primary_chg_mod}' changed"
                                )

        if symbol_matches:
            symbol_matches.sort(key=lambda m: (m[0], m[1], m[2]))
            primary_cand, primary_file, primary_mod, _, primary_conf = symbol_matches[0]
            all_reasons = [m[3] for m in symbol_matches]
            affected_tests.append(
                TestImpact(
                    node_id=test.node_id,
                    test_file=test.file_path,
                    test_name=test.test_name,
                    impact_type=ImpactType.SYMBOL,
                    matched_file=primary_file,
                    matched_module=primary_mod,
                    reasons=all_reasons,
                    matched_symbol=primary_cand,
                    confidence=primary_conf,
                )
            )
            continue

        if ast_fallback_records:
            ast_fallback_records.sort(key=lambda r: (r[0], r[1], r[2]))
            primary_fb_file, primary_fb_mod, _ = ast_fallback_records[0]
            all_fb_reasons = sorted(set(r[2] for r in ast_fallback_records))
            affected_tests.append(
                TestImpact(
                    node_id=test.node_id,
                    test_file=test.file_path,
                    test_name=test.test_name,
                    impact_type=ImpactType.MODULE,
                    matched_file=primary_fb_file,
                    matched_module=primary_fb_mod,
                    reasons=all_fb_reasons,
                    confidence=ImpactConfidence.CONSERVATIVE,
                )
            )
            continue

        if unverified_fallback_candidates:
            unverified_fallback_candidates.sort(key=lambda c: (c[0], c[1]))
            primary_unverified_file, primary_unverified_mod = unverified_fallback_candidates[0]
            affected_tests.append(
                TestImpact(
                    node_id=test.node_id,
                    test_file=test.file_path,
                    test_name=test.test_name,
                    impact_type=ImpactType.MODULE,
                    matched_file=primary_unverified_file,
                    matched_module=primary_unverified_mod,
                    reasons=sorted(set(unverified_reasons)),
                    confidence=ImpactConfidence.CONSERVATIVE,
                )
            )
            continue

        # No changed symbol matched, no uncertainty, no module-level impact -> UNAFFECTED
        unaffected_tests.append(test.node_id)

    # Sort outputs deterministically by (test_file, test_name, node_id)
    affected_tests.sort(key=lambda t: (t.test_file, t.test_name, t.node_id))
    unaffected_tests.sort()

    return ImpactAnalysisResult(
        total_discovered_tests=total_discovered,
        total_affected_tests=len(affected_tests),
        affected_tests=affected_tests,
        unaffected_tests=unaffected_tests,
    )
