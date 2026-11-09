"""Test discovery service for TestWeaver.

Discovers pytest and unittest compatible tests in Python repositories using AST parsing.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories typically ignored when searching for tests
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    "node_modules",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "site-packages",
}


class TestDiscoveryError(Exception):
    """Raised when test discovery fails unexpectedly."""


@dataclass
class DiscoveredTest:
    """Represents a discovered test function or method."""

    node_id: str
    file_path: str
    test_name: str
    line_number: int
    imported_modules: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Represents the overall test discovery results for a repository."""

    total_tests: int
    test_files_count: int
    tests: list[DiscoveredTest] = field(default_factory=list)


def _is_test_file(file_name: str) -> bool:
    """Check if a filename follows pytest test file naming conventions."""
    if not file_name.endswith(".py"):
        return False
    return file_name.startswith("test_") or file_name.endswith("_test.py")


def _is_test_function_name(name: str) -> bool:
    """Check if a function name matches pytest test naming conventions.

    By default, pytest identifies test functions prefixed with 'test_'.
    """
    return name.startswith("test_") or name == "test"


def _inherits_from_test_case(node: ast.ClassDef) -> bool:
    """Check if a class directly or indirectly inherits from a TestCase class."""
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
            return True
        if isinstance(base, ast.Name) and (base.id == "TestCase" or base.id.endswith("TestCase")):
            return True
    return False


def _is_test_class(node: ast.ClassDef) -> bool:
    """Check if a class matches pytest or unittest test class naming conventions."""
    # Standard pytest test class naming convention (e.g. TestCalculator)
    if node.name.startswith("Test") and not node.name.startswith("Test__"):
        return True

    # unittest.TestCase subclasses (e.g. CalculatorSuite(unittest.TestCase))
    if _inherits_from_test_case(node):
        return True

    return False


def _extract_imported_modules(tree: ast.AST) -> list[str]:
    """Extract module and symbol imports from an AST for dependency tracking.

    Accurately captures absolute and relative imports without inventing
    unverified absolute package paths.
    """
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level_dots = "." * node.level if node.level > 0 else ""
            if node.module:
                base = f"{level_dots}{node.module}"
                imported.add(base)
                for alias in node.names:
                    imported.add(f"{base}.{alias.name}")
                    imported.add(alias.name)
            else:
                # e.g. from . import helper (node.module is None)
                for alias in node.names:
                    if level_dots:
                        imported.add(f"{level_dots}{alias.name}")
                    imported.add(alias.name)
    return sorted(imported)


def discover_tests_in_file(
    file_path: Path,
    repo_root: Path,
) -> list[DiscoveredTest]:
    """Parse a Python file using AST and discover pytest and unittest test functions and methods."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, OSError) as exc:
        logger.warning("Could not parse file for tests %s: %s", file_path, exc)
        return []

    try:
        rel_path = file_path.relative_to(repo_root).as_posix()
    except ValueError:
        rel_path = file_path.as_posix()

    imported_modules = _extract_imported_modules(tree)
    discovered: list[DiscoveredTest] = []

    for node in tree.body:
        # Standalone test functions (def or async def)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_test_function_name(node.name):
                discovered.append(
                    DiscoveredTest(
                        node_id=f"{rel_path}::{node.name}",
                        file_path=rel_path,
                        test_name=node.name,
                        line_number=node.lineno,
                        imported_modules=imported_modules,
                    )
                )

        # Test classes (pytest classes or unittest.TestCase subclasses)
        elif isinstance(node, ast.ClassDef):
            if _is_test_class(node):
                is_unittest = _inherits_from_test_case(node)
                # Pytest ignores test classes that define __init__ (unless they are unittest.TestCase)
                has_init = any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "__init__"
                    for item in node.body
                )
                if has_init and not is_unittest:
                    continue

                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _is_test_function_name(member.name):
                            full_name = f"{node.name}::{member.name}"
                            discovered.append(
                                DiscoveredTest(
                                    node_id=f"{rel_path}::{full_name}",
                                    file_path=rel_path,
                                    test_name=full_name,
                                    line_number=member.lineno,
                                    imported_modules=imported_modules,
                                )
                            )

    return discovered


def discover_tests(
    repository_path: str | Path,
    ignored_dirs: set[str] | None = None,
) -> DiscoveryResult:
    """Discover all pytest and unittest tests in a repository directory.

    Args:
        repository_path: The root directory of the repository.
        ignored_dirs: Optional set of directory names to skip.

    Returns:
        DiscoveryResult with total test count and list of DiscoveredTest items.
    """
    root_path = Path(repository_path).expanduser().resolve()
    if not root_path.exists():
        raise TestDiscoveryError(f"Repository path does not exist: {repository_path}")
    if not root_path.is_dir():
        raise TestDiscoveryError(f"Repository path is not a directory: {repository_path}")

    skip_dirs = ignored_dirs if ignored_dirs is not None else DEFAULT_IGNORED_DIRS

    all_tests: list[DiscoveredTest] = []
    test_files_seen: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

        for filename in filenames:
            if _is_test_file(filename):
                file_path = Path(dirpath) / filename
                file_tests = discover_tests_in_file(file_path, root_path)
                test_files_seen.add(str(file_path))
                all_tests.extend(file_tests)

    return DiscoveryResult(
        total_tests=len(all_tests),
        test_files_count=len(test_files_seen),
        tests=all_tests,
    )
