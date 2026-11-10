"""Unit and integration tests for AST line-to-symbol mapping (ast_analyzer.py).

Covers:
  - Symbol extraction (functions, async functions, classes, methods, nested functions, decorators)
  - Enclosing symbol resolution (narrowest enclosing symbol, module-level fallback)
  - Symbol classification (ADDED, DELETED, MODIFIED with symbol-type identity)
  - Type transitions (function replaced by class, async replaced by normal function)
  - Non-Python files (distant changed lines, correct line ranges)
  - Extension transitions (renamed from .py to non-.py)
  - Error handling (syntax error in base or target, empty files, missing files, invalid revisions)
  - Multiple hunks in a single file
  - End-to-end repository diff symbol mapping with temporary Git repos
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from services.ast_analyzer import (
    ChangeCategory,
    CodeSymbol,
    FileSymbolChanges,
    SymbolType,
    analyze_file_diff_symbols,
    analyze_repository_symbols,
    get_node_start_line,
    parse_source_symbols,
    resolve_enclosing_symbol,
)
from services.diff_analyzer import DiffHunk, FileChangeType, FileDiff, ParsedDiff, parse_unified_diff
from services.git_service import GitServiceError, get_diff


def _cleanup_temp_dir(temp_dir: tempfile.TemporaryDirectory) -> None:
    """Safely clean up temp directory handling read-only git files on Windows."""
    try:
        temp_dir.cleanup()
    except Exception:
        import os, stat
        def _onerror(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(temp_dir.name, onerror=_onerror, ignore_errors=True)


class TestASTUnitMapping(unittest.TestCase):
    """Unit tests for AST parsing and line-to-symbol resolution without Git."""

    def test_standalone_functions(self):
        source = (
            "def add(a, b):\n"       # Line 1
            "    return a + b\n\n"   # Line 2
            "def sub(a, b):\n"       # Line 4
            "    return a - b\n"     # Line 5
        )
        res = parse_source_symbols(source)
        self.assertTrue(res.success)
        self.assertEqual(len(res.symbols), 2)

        add_sym = res.symbols[0]
        self.assertEqual(add_sym.name, "add")
        self.assertEqual(add_sym.fully_qualified_name, "add")
        self.assertEqual(add_sym.symbol_type, SymbolType.FUNCTION)
        self.assertEqual(add_sym.line_start, 1)
        self.assertEqual(add_sym.line_end, 2)

        sub_sym = res.symbols[1]
        self.assertEqual(sub_sym.name, "sub")
        self.assertEqual(sub_sym.line_start, 4)
        self.assertEqual(sub_sym.line_end, 5)

    def test_async_functions(self):
        source = (
            "async def fetch_data():\n"  # Line 1
            "    return 42\n"            # Line 2
        )
        res = parse_source_symbols(source)
        self.assertTrue(res.success)
        self.assertEqual(len(res.symbols), 1)
        sym = res.symbols[0]
        self.assertEqual(sym.name, "fetch_data")
        self.assertEqual(sym.symbol_type, SymbolType.ASYNC_FUNCTION)
        self.assertEqual(sym.line_start, 1)
        self.assertEqual(sym.line_end, 2)

    def test_classes_and_methods(self):
        source = (
            "class Calculator:\n"                # Line 1
            "    def __init__(self):\n"          # Line 2
            "        self.val = 0\n\n"           # Line 3
            "    async def async_compute(self):\n" # Line 5
            "        return self.val\n"          # Line 6
        )
        res = parse_source_symbols(source)
        self.assertTrue(res.success)
        self.assertEqual(len(res.symbols), 3)

        cls_sym = res.symbols[0]
        self.assertEqual(cls_sym.name, "Calculator")
        self.assertEqual(cls_sym.symbol_type, SymbolType.CLASS)
        self.assertEqual(cls_sym.line_start, 1)
        self.assertEqual(cls_sym.line_end, 6)

        init_sym = res.symbols[1]
        self.assertEqual(init_sym.name, "__init__")
        self.assertEqual(init_sym.fully_qualified_name, "Calculator.__init__")
        self.assertEqual(init_sym.symbol_type, SymbolType.METHOD)
        self.assertEqual(init_sym.parent_name, "Calculator")
        self.assertEqual(init_sym.line_start, 2)
        self.assertEqual(init_sym.line_end, 3)

        async_sym = res.symbols[2]
        self.assertEqual(async_sym.name, "async_compute")
        self.assertEqual(async_sym.fully_qualified_name, "Calculator.async_compute")
        self.assertEqual(async_sym.symbol_type, SymbolType.ASYNC_METHOD)
        self.assertEqual(async_sym.line_start, 5)
        self.assertEqual(async_sym.line_end, 6)

    def test_nested_functions(self):
        source = (
            "def outer(x):\n"          # Line 1
            "    def inner(y):\n"      # Line 2
            "        return x + y\n"   # Line 3
            "    return inner(1)\n"    # Line 4
        )
        res = parse_source_symbols(source)
        self.assertTrue(res.success)
        self.assertEqual(len(res.symbols), 2)

        outer_sym = res.symbols[0]
        self.assertEqual(outer_sym.name, "outer")
        self.assertEqual(outer_sym.fully_qualified_name, "outer")
        self.assertEqual(outer_sym.symbol_type, SymbolType.FUNCTION)

        inner_sym = res.symbols[1]
        self.assertEqual(inner_sym.name, "inner")
        self.assertEqual(inner_sym.fully_qualified_name, "outer.inner")
        self.assertEqual(inner_sym.symbol_type, SymbolType.FUNCTION)
        self.assertEqual(inner_sym.parent_name, "outer")
        self.assertEqual(inner_sym.line_start, 2)
        self.assertEqual(inner_sym.line_end, 3)

    def test_decorators_included_in_start_line(self):
        source = (
            "@decorator_one\n"        # Line 1
            "@decorator_two(arg=1)\n" # Line 2
            "def decorated_fn():\n"   # Line 3
            "    pass\n"              # Line 4
        )
        res = parse_source_symbols(source)
        self.assertTrue(res.success)
        sym = res.symbols[0]
        self.assertEqual(sym.name, "decorated_fn")
        self.assertEqual(sym.line_start, 1)
        self.assertEqual(sym.line_end, 4)

    def test_resolve_enclosing_symbol_narrowest(self):
        source = (
            "class Math:\n"            # Line 1
            "    def outer(self):\n"   # Line 2
            "        def inner():\n"   # Line 3
            "            return 1\n"   # Line 4
            "        return inner()\n" # Line 5
        )
        symbols = parse_source_symbols(source).symbols

        # Line 1 -> class Math
        sym_1 = resolve_enclosing_symbol(1, symbols)
        self.assertIsNotNone(sym_1)
        self.assertEqual(sym_1.fully_qualified_name, "Math")

        # Line 2 -> Math.outer
        sym_2 = resolve_enclosing_symbol(2, symbols)
        self.assertEqual(sym_2.fully_qualified_name, "Math.outer")

        # Line 4 -> innermost: Math.outer.inner
        sym_4 = resolve_enclosing_symbol(4, symbols)
        self.assertEqual(sym_4.fully_qualified_name, "Math.outer.inner")

        # Line 5 -> Math.outer
        sym_5 = resolve_enclosing_symbol(5, symbols)
        self.assertEqual(sym_5.fully_qualified_name, "Math.outer")

        # Line 10 (out of bounds) -> None (module-level)
        self.assertIsNone(resolve_enclosing_symbol(10, symbols))

    def test_syntax_error_handled_gracefully(self):
        bad_source = "def broken(\n   this is not valid python {"
        res = parse_source_symbols(bad_source)
        self.assertFalse(res.success)
        self.assertIsNotNone(res.error_message)
        self.assertIn("SyntaxError", res.error_message)
        self.assertEqual(len(res.symbols), 0)

    def test_empty_source_returns_empty_symbols_success(self):
        res = parse_source_symbols("")
        self.assertTrue(res.success)
        self.assertEqual(len(res.symbols), 0)


class TestSymbolClassification(unittest.TestCase):
    """Unit tests for analyze_file_diff_symbols change classification."""

    def test_modified_symbol_aggregation(self):
        base_src = "def calculate(a, b):\n    return a + b\n"
        target_src = "def calculate(a, b):\n    # updated\n    return a * b\n"

        file_diff = FileDiff(
            old_path="calc.py",
            new_path="calc.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=2,
                    old_count=1,
                    new_start=2,
                    new_count=2,
                    deleted_lines=[2],
                    added_lines=[2, 3],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "calculate")
        self.assertEqual(sym.change_category, ChangeCategory.MODIFIED)
        self.assertEqual(sym.line_start, 1)
        self.assertEqual(sym.line_end, 3)
        self.assertEqual(sym.changed_lines, [2, 3])

    def test_added_symbol_classification(self):
        base_src = "def existing():\n    pass\n"
        target_src = "def existing():\n    pass\n\ndef new_func():\n    return 42\n"

        file_diff = FileDiff(
            old_path="calc.py",
            new_path="calc.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=2,
                    old_count=0,
                    new_start=4,
                    new_count=2,
                    deleted_lines=[],
                    added_lines=[4, 5],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "new_func")
        self.assertEqual(sym.change_category, ChangeCategory.ADDED)
        self.assertEqual(sym.line_start, 4)
        self.assertEqual(sym.line_end, 5)
        self.assertEqual(sym.changed_lines, [4, 5])

    def test_deleted_symbol_classification(self):
        base_src = "def keep():\n    pass\n\ndef remove_me():\n    return 1\n"
        target_src = "def keep():\n    pass\n"

        file_diff = FileDiff(
            old_path="calc.py",
            new_path="calc.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=4,
                    old_count=2,
                    new_start=2,
                    new_count=0,
                    deleted_lines=[4, 5],
                    added_lines=[],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "remove_me")
        self.assertEqual(sym.change_category, ChangeCategory.DELETED)
        # Deleted symbol line range refers to base revision
        self.assertEqual(sym.line_start, 4)
        self.assertEqual(sym.line_end, 5)
        self.assertEqual(sym.changed_lines, [4, 5])

    def test_module_level_changes(self):
        base_src = "import os\n\ndef foo():\n    pass\n"
        target_src = "import os\nimport sys\n\ndef foo():\n    pass\n"

        file_diff = FileDiff(
            old_path="module.py",
            new_path="module.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=1,
                    new_start=1,
                    new_count=2,
                    deleted_lines=[],
                    added_lines=[2],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "<module>")
        self.assertEqual(sym.symbol_type, SymbolType.MODULE)
        self.assertEqual(sym.change_category, ChangeCategory.MODIFIED)
        self.assertEqual(sym.changed_lines, [2])

    def test_syntax_error_fallback_to_module(self):
        base_src = "def good():\n    return 1\n"
        target_src = "def good():\n    syntax error !!\n"

        file_diff = FileDiff(
            old_path="test.py",
            new_path="test.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=2,
                    old_count=1,
                    new_start=2,
                    new_count=1,
                    deleted_lines=[2],
                    added_lines=[2],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertFalse(changes.target_parse_success)
        self.assertIn("SyntaxError", changes.target_parse_error)
        self.assertEqual(len(changes.changed_symbols), 1)
        self.assertEqual(changes.changed_symbols[0].symbol_name, "<module>")

    # -------------------------------------------------------------------------
    # Correction 1: Symbol identity includes symbol type
    # -------------------------------------------------------------------------

    def test_function_replaced_by_class_with_same_name(self):
        """A function replaced by a class of the same name produces DELETED and ADDED records."""
        base_src = "def Processor():\n    pass\n"
        target_src = "class Processor:\n    pass\n"

        file_diff = FileDiff(
            old_path="proc.py",
            new_path="proc.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=2,
                    new_start=1,
                    new_count=2,
                    deleted_lines=[1, 2],
                    added_lines=[1, 2],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 2)

        deleted_sym = next(s for s in changes.changed_symbols if s.change_category == ChangeCategory.DELETED)
        self.assertEqual(deleted_sym.symbol_name, "Processor")
        self.assertEqual(deleted_sym.symbol_type, SymbolType.FUNCTION)
        self.assertEqual(deleted_sym.changed_lines, [1, 2])

        added_sym = next(s for s in changes.changed_symbols if s.change_category == ChangeCategory.ADDED)
        self.assertEqual(added_sym.symbol_name, "Processor")
        self.assertEqual(added_sym.symbol_type, SymbolType.CLASS)
        self.assertEqual(added_sym.changed_lines, [1, 2])

    def test_async_function_replaced_by_normal_function(self):
        """An async function replaced by a regular function produces DELETED and ADDED records."""
        base_src = "async def fetch_data():\n    return 1\n"
        target_src = "def fetch_data():\n    return 1\n"

        file_diff = FileDiff(
            old_path="fetch.py",
            new_path="fetch.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=2,
                    new_start=1,
                    new_count=2,
                    deleted_lines=[1],
                    added_lines=[1],
                )
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 2)

        deleted_sym = next(s for s in changes.changed_symbols if s.change_category == ChangeCategory.DELETED)
        self.assertEqual(deleted_sym.symbol_name, "fetch_data")
        self.assertEqual(deleted_sym.symbol_type, SymbolType.ASYNC_FUNCTION)
        self.assertEqual(deleted_sym.changed_lines, [1])

        added_sym = next(s for s in changes.changed_symbols if s.change_category == ChangeCategory.ADDED)
        self.assertEqual(added_sym.symbol_name, "fetch_data")
        self.assertEqual(added_sym.symbol_type, SymbolType.FUNCTION)
        self.assertEqual(added_sym.changed_lines, [1])

    # -------------------------------------------------------------------------
    # Correction 2: Non-Python line ranges
    # -------------------------------------------------------------------------

    def test_non_python_file_with_distant_changed_lines(self):
        """A non-Python file where only distant lines (e.g. 20 and 30) change preserves file length."""
        # 50 lines of markdown
        lines = [f"# Line {i}" for i in range(1, 51)]
        target_src = "\n".join(lines) + "\n"
        base_src = target_src

        file_diff = FileDiff(
            old_path="README.md",
            new_path="README.md",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(old_start=20, old_count=1, new_start=20, new_count=1, deleted_lines=[20], added_lines=[20]),
                DiffHunk(old_start=30, old_count=1, new_start=30, new_count=1, deleted_lines=[30], added_lines=[30]),
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "<module>")
        self.assertEqual(sym.symbol_type, SymbolType.MODULE)
        self.assertEqual(sym.change_category, ChangeCategory.MODIFIED)
        self.assertEqual(sym.line_start, 1)
        self.assertEqual(sym.line_end, 50, "Line end must reflect the file line count, not changed line count")
        self.assertEqual(sym.changed_lines, [20, 30])

    def test_non_python_file_source_unavailable_fallback(self):
        """When source is unavailable for a non-Python file, line range bounds the changed lines."""
        file_diff = FileDiff(
            old_path="config.yaml",
            new_path="config.yaml",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(old_start=20, old_count=1, new_start=20, new_count=1, deleted_lines=[20], added_lines=[20]),
                DiffHunk(old_start=30, old_count=1, new_start=30, new_count=1, deleted_lines=[30], added_lines=[30]),
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_source=None, target_source=None)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.line_start, 20)
        self.assertEqual(sym.line_end, 30)
        self.assertEqual(sym.changed_lines, [20, 30])

    # -------------------------------------------------------------------------
    # Additional edge cases: Base syntax error & Multiple hunks & Extension change
    # -------------------------------------------------------------------------

    def test_syntax_error_in_base_revision(self):
        """A syntax error in the base revision falls back to module-level without false additions."""
        base_src = "def bad_syntax(\n  unclosed bracket {"
        target_src = "def valid():\n    return 42\n"

        file_diff = FileDiff(
            old_path="test.py",
            new_path="test.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(old_start=1, old_count=2, new_start=1, new_count=2, deleted_lines=[1, 2], added_lines=[1, 2])
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertFalse(changes.base_parse_success)
        self.assertIn("SyntaxError", changes.base_parse_error)
        self.assertTrue(changes.target_parse_success)
        self.assertEqual(len(changes.changed_symbols), 1)
        self.assertEqual(changes.changed_symbols[0].symbol_name, "<module>")

    def test_multiple_changed_hunks_mapped_to_correct_symbols(self):
        """Multiple hunks in a single file correctly map to their respective symbols."""
        base_src = (
            "def func_a():\n"      # Line 1
            "    return 'a'\n\n"    # Line 2
            "def func_b():\n"      # Line 4
            "    return 'b'\n"      # Line 5
        )
        target_src = (
            "def func_a():\n"      # Line 1
            "    return 'A'\n\n"    # Line 2
            "def func_b():\n"      # Line 4
            "    return 'B'\n"      # Line 5
        )

        file_diff = FileDiff(
            old_path="funcs.py",
            new_path="funcs.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                DiffHunk(old_start=2, old_count=1, new_start=2, new_count=1, deleted_lines=[2], added_lines=[2]),
                DiffHunk(old_start=5, old_count=1, new_start=5, new_count=1, deleted_lines=[5], added_lines=[5]),
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(len(changes.changed_symbols), 2)

        sym_a = next(s for s in changes.changed_symbols if s.symbol_name == "func_a")
        self.assertEqual(sym_a.changed_lines, [2])
        self.assertEqual(sym_a.change_category, ChangeCategory.MODIFIED)

        sym_b = next(s for s in changes.changed_symbols if s.symbol_name == "func_b")
        self.assertEqual(sym_b.changed_lines, [5])
        self.assertEqual(sym_b.change_category, ChangeCategory.MODIFIED)

    def test_renamed_python_to_non_python_extension(self):
        """Renaming a Python file to a non-Python extension falls back to module-level tracking."""
        base_src = "def old_code():\n    return True\n"
        target_src = "def old_code():\n    return True\n# now a text file\n"

        file_diff = FileDiff(
            old_path="script.py",
            new_path="script.txt",
            change_type=FileChangeType.RENAMED,
            hunks=[
                DiffHunk(old_start=2, old_count=0, new_start=3, new_count=1, deleted_lines=[], added_lines=[3])
            ],
        )

        changes = analyze_file_diff_symbols(file_diff, base_src, target_src)
        self.assertEqual(changes.change_type, FileChangeType.RENAMED)
        self.assertEqual(len(changes.changed_symbols), 1)
        sym = changes.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "<module>")
        self.assertEqual(sym.symbol_type, SymbolType.MODULE)
        self.assertEqual(sym.changed_lines, [3])


class TestASTRepositoryIntegration(unittest.TestCase):
    """Integration tests running Git operations across revisions in temporary repos."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name).resolve()
        self.repo_str = str(self.repo_path)

        subprocess.run(["git", "init", self.repo_str], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", self.repo_str, "config", "user.email", "test@testweaver.internal"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", self.repo_str, "config", "user.name", "TestWeaver Test"],
            check=True,
            capture_output=True,
        )

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)
        super().tearDown()

    def _commit(self, msg: str) -> str:
        subprocess.run(["git", "-C", self.repo_str, "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", self.repo_str, "commit", "-m", msg],
            check=True,
            capture_output=True,
        )
        rev = subprocess.run(
            ["git", "-C", self.repo_str, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return rev.stdout.strip()

    def test_repository_modify_function_and_add_function(self):
        # Base commit: calculator with add
        calc_path = self.repo_path / "calculator.py"
        calc_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        base_rev = self._commit("Initial commit with add")

        # Target commit: modify add, add multiply
        calc_path.write_text(
            "def add(a, b):\n"
            "    # validated add\n"
            "    return int(a) + int(b)\n\n"
            "def multiply(a, b):\n"
            "    return a * b\n",
            encoding="utf-8",
        )
        target_rev = self._commit("Modify add and add multiply")

        diff_text = get_diff(self.repo_str, base_rev, target_rev)
        parsed_diff = parse_unified_diff(diff_text)

        results = analyze_repository_symbols(self.repo_str, base_rev, target_rev, parsed_diff)
        self.assertEqual(len(results), 1)

        file_changes = results[0]
        self.assertEqual(file_changes.file_path, "calculator.py")
        self.assertTrue(file_changes.base_parse_success)
        self.assertTrue(file_changes.target_parse_success)

        names = {s.symbol_name: s for s in file_changes.changed_symbols}
        self.assertIn("add", names)
        self.assertEqual(names["add"].change_category, ChangeCategory.MODIFIED)

        self.assertIn("multiply", names)
        self.assertEqual(names["multiply"].change_category, ChangeCategory.ADDED)

    def test_repository_renamed_file(self):
        # Base commit: old_calc.py
        orig_file = self.repo_path / "old_calc.py"
        orig_file.write_text("def compute():\n    return 10\n", encoding="utf-8")
        base_rev = self._commit("Add old_calc.py")

        # Target commit: renamed to new_calc.py and modified compute()
        subprocess.run(
            ["git", "-C", self.repo_str, "mv", "old_calc.py", "new_calc.py"],
            check=True,
            capture_output=True,
        )
        new_file = self.repo_path / "new_calc.py"
        new_file.write_text("def compute():\n    return 20\n", encoding="utf-8")
        target_rev = self._commit("Rename and update compute")

        # Create a parsed diff representing the renamed file
        parsed_diff = ParsedDiff(
            files=[
                FileDiff(
                    old_path="old_calc.py",
                    new_path="new_calc.py",
                    change_type=FileChangeType.RENAMED,
                    hunks=[
                        DiffHunk(
                            old_start=2,
                            old_count=1,
                            new_start=2,
                            new_count=1,
                            deleted_lines=[2],
                            added_lines=[2],
                        )
                    ],
                )
            ]
        )

        results = analyze_repository_symbols(self.repo_str, base_rev, target_rev, parsed_diff)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.change_type, FileChangeType.RENAMED)
        self.assertEqual(res.old_path, "old_calc.py")
        self.assertEqual(res.new_path, "new_calc.py")

        self.assertEqual(len(res.changed_symbols), 1)
        sym = res.changed_symbols[0]
        self.assertEqual(sym.symbol_name, "compute")
        self.assertEqual(sym.change_category, ChangeCategory.MODIFIED)
        self.assertEqual(sym.changed_lines, [2])

    def test_repository_invalid_revision_raises_git_service_error(self):
        calc_path = self.repo_path / "dummy.py"
        calc_path.write_text("x = 1\n", encoding="utf-8")
        base_rev = self._commit("Initial")

        with self.assertRaises(GitServiceError):
            analyze_repository_symbols(self.repo_str, base_rev, "nonexistent_rev_123", ParsedDiff())


if __name__ == "__main__":
    unittest.main()
