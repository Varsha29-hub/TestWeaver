"""Unit tests for the Module-Level Test Impact Analyzer (services/impact_analyzer.py).

Covers:
  1. Exact module match.
  2. No module match.
  3. Multiple tests importing the same changed module.
  4. Multiple changed modules matching one test without duplicate results.
  5. Renamed source file matching its old module path.
  6. Renamed source file matching its new module path.
  7. Deleted Python file.
  8. Added Python file.
  9. Non-Python files ignored.
  10. Windows path separators.
  11. Package __init__.py mapping.
  12. Imported submodule matching a changed parent module.
  13. Imported symbol matching its parent module.
  14. Similar module names do not produce false matches.
  15. Empty symbol changes.
  16. Empty discovery results.
  17. Test with no imports.
  18. Helper functions (file_to_module_path, is_module_affected).
  19. Regression: Relative imports are rejected from naive dot-stripping matching in Stage 1.
  20. Regression: Canonical module paths are preserved; no heuristic stripping of src. or lib.
"""

from typing import Optional
import unittest

from services.ast_analyzer import (
    ChangeCategory,
    ChangedSymbol,
    FileSymbolChanges,
    SymbolType,
)
from services.diff_analyzer import FileChangeType
from services.impact_analyzer import (
    ImpactAnalysisResult,
    ImpactConfidence,
    ImpactType,
    TestImpact,
    VerificationStatus,
    analyze_test_impact,
    file_to_module_path,
    get_importable_module_paths,
    is_module_affected,
    verify_symbol_in_source,
)
from services.test_discovery import DiscoveredTest, DiscoveryResult


class TestImpactAnalyzerHelpers(unittest.TestCase):
    """Unit tests for path-to-module conversion and module matching helpers."""

    def test_file_to_module_path_standard(self):
        self.assertEqual(file_to_module_path("src/calculator.py"), "src.calculator")
        self.assertEqual(file_to_module_path("src/utils/string_helpers.py"), "src.utils.string_helpers")
        self.assertEqual(file_to_module_path("calculator.py"), "calculator")
        self.assertEqual(file_to_module_path("lib/helpers.py"), "lib.helpers")

    def test_file_to_module_path_windows_separators(self):
        self.assertEqual(file_to_module_path("src\\utils\\string_helpers.py"), "src.utils.string_helpers")
        self.assertEqual(file_to_module_path("src\\calculator.py"), "src.calculator")

    def test_file_to_module_path_package_init(self):
        self.assertEqual(file_to_module_path("src/utils/__init__.py"), "src.utils")
        self.assertEqual(file_to_module_path("pkg/__init__.py"), "pkg")
        self.assertIsNone(file_to_module_path("__init__.py"))

    def test_file_to_module_path_non_python(self):
        self.assertIsNone(file_to_module_path("README.md"))
        self.assertIsNone(file_to_module_path("config.json"))
        self.assertIsNone(file_to_module_path("script.py.bak"))
        self.assertIsNone(file_to_module_path(""))

    def test_get_importable_module_paths_canonical_only(self):
        """get_importable_module_paths retains canonical paths without stripping src. or lib."""
        paths = get_importable_module_paths("src/calculator.py")
        self.assertEqual(paths, ["src.calculator"])
        self.assertNotIn("calculator", paths)

        lib_paths = get_importable_module_paths("lib/helpers.py")
        self.assertEqual(lib_paths, ["lib.helpers"])
        self.assertNotIn("helpers", lib_paths)

    def test_is_module_affected_exact(self):
        self.assertTrue(is_module_affected("src.calculator", "src.calculator"))
        self.assertTrue(is_module_affected("calculator", "calculator"))

    def test_is_module_affected_submodule_and_symbol(self):
        self.assertTrue(is_module_affected("src.calculator", "src.calculator.add"))
        self.assertTrue(is_module_affected("src.utils", "src.utils.string_helpers"))
        self.assertTrue(is_module_affected("src.utils", "src.utils.string_helpers.strip"))

    def test_is_module_affected_no_false_substring_match(self):
        self.assertFalse(is_module_affected("src.calculator", "src.calculator_extra"))
        self.assertFalse(is_module_affected("src.calculator", "src.calculator_extra.add"))
        self.assertFalse(is_module_affected("calc", "calculator"))

    def test_is_module_affected_relative_import_unmatched_in_stage_1(self):
        """Relative imports with leading dots are not matched via naive dot-stripping."""
        self.assertFalse(is_module_affected("src.calculator", ".src.calculator"))
        self.assertFalse(is_module_affected("src.calculator", "..calculator"))
        self.assertFalse(is_module_affected("calculator", ".calculator"))


class TestAnalyzeTestImpact(unittest.TestCase):
    """Unit tests for analyze_test_impact across all required scenarios."""

    def _make_test(self, node_id: str, imports: list[str]) -> DiscoveredTest:
        file_path, test_name = node_id.split("::", 1) if "::" in node_id else (node_id, "test_func")
        return DiscoveredTest(
            node_id=node_id,
            file_path=file_path,
            test_name=test_name,
            line_number=10,
            imported_modules=imports,
        )

    def _make_discovery_result(self, tests: list[DiscoveredTest]) -> DiscoveryResult:
        files = {t.file_path for t in tests}
        return DiscoveryResult(
            total_tests=len(tests),
            test_files_count=len(files),
            tests=tests,
        )

    def test_1_exact_module_match(self):
        """A test that imports the exact changed module is marked as affected."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_calc.py::test_add", ["src.calculator"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_discovered_tests, 1)
        self.assertEqual(res.total_affected_tests, 1)
        self.assertEqual(len(res.affected_tests), 1)
        self.assertEqual(res.unaffected_tests, [])

        impact = res.affected_tests[0]
        self.assertEqual(impact.node_id, "tests/test_calc.py::test_add")
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.matched_file, "src/calculator.py")
        self.assertEqual(impact.matched_module, "src.calculator")
        self.assertTrue(any("src.calculator" in r for r in impact.reasons))

    def test_2_no_module_match(self):
        """A test whose imports do not match any changed module is marked unaffected."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_users.py::test_login", ["src.users", "pytest"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.affected_tests, [])
        self.assertEqual(res.unaffected_tests, ["tests/test_users.py::test_login"])

    def test_3_multiple_tests_importing_same_changed_module(self):
        """Multiple tests importing the same module are all identified as affected."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test_1 = self._make_test("tests/test_calc.py::test_add", ["src.calculator"])
        test_2 = self._make_test("tests/test_calc.py::test_sub", ["src.calculator"])
        discovery = self._make_discovery_result([test_1, test_2])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 2)
        node_ids = [t.node_id for t in res.affected_tests]
        self.assertIn("tests/test_calc.py::test_add", node_ids)
        self.assertIn("tests/test_calc.py::test_sub", node_ids)
        self.assertEqual(res.unaffected_tests, [])

    def test_4_multiple_changed_modules_matching_one_test_deduplicated(self):
        """A test importing multiple changed modules appears once with reasons for each."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            ),
            FileSymbolChanges(
                file_path="src/formatter.py",
                old_path="src/formatter.py",
                new_path="src/formatter.py",
                change_type=FileChangeType.MODIFIED,
            ),
        ]
        test = self._make_test("tests/test_calc.py::test_all", ["src.calculator", "src.formatter"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        self.assertEqual(len(res.affected_tests), 1)

        impact = res.affected_tests[0]
        self.assertEqual(len(impact.reasons), 2)
        reasons_text = " ".join(impact.reasons)
        self.assertIn("src.calculator", reasons_text)
        self.assertIn("src.formatter", reasons_text)

    def test_5_renamed_source_file_matching_old_module_path(self):
        """A test importing the old module name matches a renamed file."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="new_calc.py",
                old_path="old_calc.py",
                new_path="new_calc.py",
                change_type=FileChangeType.RENAMED,
            )
        ]
        test = self._make_test("tests/test_legacy.py::test_old", ["old_calc"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.matched_file, "old_calc.py")
        self.assertEqual(impact.matched_module, "old_calc")
        self.assertTrue(any("renamed from 'old_calc.py'" in r for r in impact.reasons))

    def test_6_renamed_source_file_matching_new_module_path(self):
        """A test importing the new module name matches a renamed file."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="new_calc.py",
                old_path="old_calc.py",
                new_path="new_calc.py",
                change_type=FileChangeType.RENAMED,
            )
        ]
        test = self._make_test("tests/test_modern.py::test_new", ["new_calc"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.matched_file, "new_calc.py")
        self.assertEqual(impact.matched_module, "new_calc")
        self.assertTrue(any("renamed to 'new_calc.py'" in r for r in impact.reasons))

    def test_7_deleted_python_file(self):
        """Tests importing a deleted Python file are identified as affected."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/legacy_service.py",
                old_path="src/legacy_service.py",
                new_path=None,
                change_type=FileChangeType.DELETED,
            )
        ]
        test = self._make_test("tests/test_svc.py::test_legacy", ["src.legacy_service"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.matched_file, "src/legacy_service.py")
        self.assertTrue(any("deleted file" in r for r in impact.reasons))

    def test_8_added_python_file(self):
        """Tests importing a newly added Python file are identified as affected."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/new_feature.py",
                old_path=None,
                new_path="src/new_feature.py",
                change_type=FileChangeType.ADDED,
            )
        ]
        test = self._make_test("tests/test_feature.py::test_new_feature", ["src.new_feature"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.matched_file, "src/new_feature.py")
        self.assertTrue(any("added file" in r for r in impact.reasons))

    def test_9_non_python_files_ignored(self):
        """Changes to non-Python files (e.g. README.md, config.yaml) are ignored."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="README.md",
                old_path="README.md",
                new_path="README.md",
                change_type=FileChangeType.MODIFIED,
            ),
            FileSymbolChanges(
                file_path="config.yaml",
                old_path="config.yaml",
                new_path="config.yaml",
                change_type=FileChangeType.MODIFIED,
            ),
        ]
        test = self._make_test("tests/test_app.py::test_app", ["src.app"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_app.py::test_app"])

    def test_10_windows_path_separators(self):
        """Changed files with Windows path separators match test imports cleanly."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src\\utils\\string_helpers.py",
                old_path="src\\utils\\string_helpers.py",
                new_path="src\\utils\\string_helpers.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_utils.py::test_str", ["src.utils.string_helpers"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)

    def test_11_package_init_py_mapping(self):
        """A changed __init__.py associates with the enclosing package module name."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/utils/__init__.py",
                old_path="src/utils/__init__.py",
                new_path="src/utils/__init__.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_utils.py::test_pkg", ["src.utils"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        self.assertEqual(res.affected_tests[0].matched_module, "src.utils")

    def test_12_imported_submodule_matching_changed_parent_module(self):
        """Importing a submodule matches when its parent package module changed."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/utils/__init__.py",
                old_path="src/utils/__init__.py",
                new_path="src/utils/__init__.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_sub.py::test_submod", ["src.utils.string_helpers"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)
        self.assertTrue(any("src.utils" in r for r in res.affected_tests[0].reasons))

    def test_13_imported_symbol_matching_parent_module(self):
        """Importing a symbol (e.g. from src.calculator import add) matches the changed module."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_calc.py::test_fn", ["src.calculator.add"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 1)

    def test_14_similar_module_names_do_not_produce_false_matches(self):
        """Similar module names like src.calculator_extra do NOT match src.calculator."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_extra.py::test_extra", ["src.calculator_extra"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_extra.py::test_extra"])

    def test_15_empty_symbol_changes(self):
        """Empty symbol change list leaves all discovered tests unaffected."""
        test = self._make_test("tests/test_calc.py::test_add", ["src.calculator"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact([], discovery)
        self.assertEqual(res.total_discovered_tests, 1)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.affected_tests, [])
        self.assertEqual(res.unaffected_tests, ["tests/test_calc.py::test_add"])

    def test_16_empty_discovery_results(self):
        """Empty discovery result produces zero affected and zero unaffected tests."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        discovery = DiscoveryResult(total_tests=0, test_files_count=0, tests=[])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_discovered_tests, 0)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.affected_tests, [])
        self.assertEqual(res.unaffected_tests, [])

    def test_17_test_with_no_imports(self):
        """A test with no imports is never affected and remains in unaffected_tests."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_pure.py::test_pure", [])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_pure.py::test_pure"])

    # -------------------------------------------------------------------------
    # Code review regression tests
    # -------------------------------------------------------------------------

    def test_regression_relative_import_unmatched_in_stage_1(self):
        """Relative imports (.calculator, ..utils) do not match via naive dot-stripping."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        # Test importing relative module without package anchor
        test = self._make_test("tests/test_rel.py::test_rel", [".calculator", "..calculator"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_rel.py::test_rel"])

    def test_regression_source_root_assumption_rejected(self):
        """Canonical paths are preserved without heuristic stripping of src. or lib."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
            ),
            FileSymbolChanges(
                file_path="lib/helpers.py",
                old_path="lib/helpers.py",
                new_path="lib/helpers.py",
                change_type=FileChangeType.MODIFIED,
            ),
        ]
        # Test importing 'calculator' (bare) should NOT match 'src/calculator.py'
        # because 'src.' is not stripped without explicit repository configuration.
        test_bare_calc = self._make_test("tests/test_calc.py::test_bare", ["calculator"])
        test_bare_lib = self._make_test("tests/test_lib.py::test_bare", ["helpers"])
        discovery = self._make_discovery_result([test_bare_calc, test_bare_lib])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertIn("tests/test_calc.py::test_bare", res.unaffected_tests)
        self.assertIn("tests/test_lib.py::test_bare", res.unaffected_tests)

    def test_regression_different_source_roots_do_not_collide(self):
        """Modules in different root directories (e.g. src.calc vs lib.calc) do not collide."""
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
            )
        ]
        test = self._make_test("tests/test_calc.py::test_lib", ["lib.calc"])
        discovery = self._make_discovery_result([test])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_calc.py::test_lib"])


class TestStage2SymbolImpactIntegration(unittest.TestCase):
    """Integration test suite for Stage 2 symbol-level test impact analysis."""

    def _make_test(self, node_id: str, imports: list[str] = None) -> DiscoveredTest:
        file_path, test_name = node_id.split("::", 1) if "::" in node_id else (node_id, "test_func")
        return DiscoveredTest(
            node_id=node_id,
            file_path=file_path,
            test_name=test_name,
            line_number=1,
            imported_modules=imports or [],
        )

    def _make_discovery_result(self, tests: list[DiscoveredTest]) -> DiscoveryResult:
        files = {t.file_path for t in tests}
        return DiscoveryResult(
            total_tests=len(tests),
            test_files_count=len(files),
            tests=tests,
        )

    def _make_source_provider(
        self,
        base_files: Optional[dict[str, str]] = None,
        target_files: Optional[dict[str, str]] = None,
    ):
        base = base_files or {}
        target = target_files or {}

        def provider(rev: str, path: str) -> Optional[str]:
            if rev == "base":
                return base.get(path)
            elif rev == "target":
                return target.get(path)
            return None

        return provider

    def _make_test_source_provider(self, test_files: dict[str, str]):
        def provider(path: str) -> Optional[str]:
            return test_files.get(path)

        return provider

    def test_verify_symbol_in_source_statuses(self):
        """Verify the 5 mutually exclusive verification statuses."""
        source = """
def outer():
    class Calculator:
        def solve(self):
            return 42
"""
        # 1. FOUND
        status, node = verify_symbol_in_source(source, "outer.Calculator.solve")
        self.assertEqual(status, VerificationStatus.FOUND)
        self.assertIsNotNone(node)

        # 2. CONFIRMED_ABSENT
        status, node = verify_symbol_in_source(source, "outer.Calculator.missing")
        self.assertEqual(status, VerificationStatus.CONFIRMED_ABSENT)
        self.assertIsNone(node)

        # 3. SCOPE_NOT_FOUND
        status, node = verify_symbol_in_source(source, "outer.MissingClass.solve")
        self.assertEqual(status, VerificationStatus.SCOPE_NOT_FOUND)
        self.assertIsNone(node)

        # 4. SOURCE_UNAVAILABLE
        status, node = verify_symbol_in_source(None, "outer.Calculator.solve")
        self.assertEqual(status, VerificationStatus.SOURCE_UNAVAILABLE)
        self.assertIsNone(node)

        # 5. SYNTAX_ERROR
        bad_source = "def broken(:"
        status, node = verify_symbol_in_source(bad_source, "broken")
        self.assertEqual(status, VerificationStatus.SYNTAX_ERROR)
        self.assertIsNone(node)

    def test_stage2_1_direct_function_match(self):
        """Test calls add() when src.calc.add was modified -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import add

def test_addition():
    assert add(1, 2) == 3
"""
        test = self._make_test("tests/test_calc.py::test_addition", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def add(a, b):\n    return a + b\n"},
            target_files={"src/calc.py": "def add(a, b):\n    return a + b + 0\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        self.assertEqual(res.unaffected_tests, [])
        impact = res.affected_tests[0]
        self.assertEqual(impact.node_id, "tests/test_calc.py::test_addition")
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.add")
        self.assertEqual(impact.matched_file, "src/calc.py")
        self.assertEqual(impact.matched_module, "src.calc")

    def test_stage2_2_class_instantiation_resolved_class_change(self):
        """calc = Calculator(); calc.solve() when Calculator class changed -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import Calculator

def test_calc_instantiation():
    calc = Calculator()
    calc.solve()
"""
        test = self._make_test("tests/test_calc.py::test_calc_instantiation", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="Calculator",
                        fully_qualified_name="Calculator",
                        symbol_type=SymbolType.CLASS,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=5,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "class Calculator:\n    def solve(self):\n        return 1\n"},
            target_files={"src/calc.py": "class Calculator:\n    # updated\n    def solve(self):\n        return 1\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.Calculator")

    def test_stage2_3_modified_method_granularity(self):
        """calc.solve() called when Calculator.solve was modified -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import Calculator

def test_method():
    calc = Calculator()
    calc.solve()
"""
        test = self._make_test("tests/test_calc.py::test_method", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="Calculator.solve",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=4,
                        line_end=5,
                        changed_lines=[5],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 1\n"},
            target_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 2\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.Calculator.solve")

    def test_stage2_4_unchanged_method_same_class(self):
        """calc.clear() called when only Calculator.solve was modified -> UNAFFECTED."""
        test_code = """
from src.calc import Calculator

def test_clear():
    calc = Calculator()
    calc.clear()
"""
        test = self._make_test("tests/test_calc.py::test_clear", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="Calculator.solve",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=4,
                        line_end=5,
                        changed_lines=[5],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 1\n"},
            target_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 2\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.affected_tests, [])
        self.assertEqual(res.unaffected_tests, ["tests/test_calc.py::test_clear"])

    def test_stage2_5_precise_relative_import_disjoint(self):
        """Test at tests/unit/test_foo.py imports .helpers; changed file is services/calculator.py -> UNAFFECTED."""
        test_code = """
from .helpers import dummy

def test_foo():
    dummy()
"""
        test = self._make_test("tests/unit/test_foo.py::test_foo", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="services/calculator.py",
                old_path="services/calculator.py",
                new_path="services/calculator.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="calculate",
                        fully_qualified_name="calculate",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/unit/test_foo.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/unit/test_foo.py::test_foo"])

    def test_stage2_6_precise_relative_import_affected(self):
        """Test at tests/unit/test_foo.py imports ..calculator; changed module is tests.calculator -> MODULE, CONSERVATIVE."""
        test_code = """
from ..calculator import calculate

def test_rel():
    calculate()
"""
        test = self._make_test("tests/unit/test_foo.py::test_rel", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="tests/calculator.py",
                old_path="tests/calculator.py",
                new_path="tests/calculator.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/unit/test_foo.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "tests.calculator")
        self.assertTrue(any("resolves to affected module 'tests.calculator'" in r for r in impact.reasons))

    def test_stage2_7_syntactically_invalid_relative_import(self):
        """Test at tests/test_foo.py uses 'from ..services import calc' (ValueError) -> UNAFFECTED."""
        test_code = """
from ..services import calc

def test_invalid():
    calc()
"""
        test = self._make_test("tests/test_foo.py::test_invalid", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="services/calc.py",
                old_path="services/calc.py",
                new_path="services/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/test_foo.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_foo.py::test_invalid"])

    def test_stage2_8_unanchored_relative_import_fallback(self):
        """Test file outside package context contains relative import while Python changes exist -> MODULE, CONSERVATIVE."""
        test_code = """
from .helpers import calc

def test_unanchored():
    calc()
"""
        test = self._make_test("test_foo.py::test_unanchored", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="services/calculator.py",
                old_path="services/calculator.py",
                new_path="services/calculator.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"test_foo.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("outside package context" in r for r in impact.reasons))

    def test_stage2_9_uncertainty_fallback_dynamic_reflection(self):
        """Test uses getattr(calc, 'solve')() while files changed -> MODULE, CONSERVATIVE."""
        test_code = """
import src.calc as calc

def test_dynamic():
    fn = getattr(calc, "solve")
    fn()
"""
        test = self._make_test("tests/test_calc.py::test_dynamic", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("unresolved dynamic uncertainty" in r for r in impact.reasons))

    def test_stage2_10_uncertainty_fallback_unknown_factory(self):
        """Test uses calc = create_service(); calc.solve() -> MODULE, CONSERVATIVE."""
        test_code = """
import src.calc

def test_factory():
    calc = create_service()
    calc.solve()
"""
        test = self._make_test("tests/test_calc.py::test_factory", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("unresolved dynamic uncertainty" in r for r in impact.reasons))

    def test_stage2_11_unparseable_test_file_fallback(self):
        """Test file has SyntaxError while Python changes exist in repo -> MODULE, CONSERVATIVE."""
        test_code = """def test_syntax_error(\n    assert True\n"""
        test = self._make_test("tests/test_broken.py::test_syntax_error", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/test_broken.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("syntax errors" in r for r in impact.reasons))

    def test_stage2_12_unparseable_test_file_zero_changes(self):
        """Test file has SyntaxError but zero Python changes in repo -> UNAFFECTED."""
        test_code = """def test_syntax_error(\n    assert True\n"""
        test = self._make_test("tests/test_broken.py::test_syntax_error", [])
        discovery = self._make_discovery_result([test])

        test_prov = self._make_test_source_provider({"tests/test_broken.py": test_code})

        res = analyze_test_impact(
            [],
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 0)
        self.assertEqual(res.unaffected_tests, ["tests/test_broken.py::test_syntax_error"])

    def test_stage2_13_multi_scope_nested_ast_traversal(self):
        """Class inside function outer.Calculator.solve modified -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import outer

def test_nested():
    c = outer.Calculator()
    c.solve()
"""
        test = self._make_test("tests/test_calc.py::test_nested", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="outer.Calculator.solve",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=3,
                        line_end=4,
                        changed_lines=[4],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def outer():\n    class Calculator:\n        def solve(self):\n            return 1\n"},
            target_files={"src/calc.py": "def outer():\n    class Calculator:\n        def solve(self):\n            return 2\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.outer.Calculator.solve")

    def test_stage2_14_added_method_verification(self):
        """Test calls newly added method; verified in target AST and absent from base AST -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import Calculator

def test_added():
    c = Calculator()
    c.solve()
"""
        test = self._make_test("tests/test_calc.py::test_added", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="Calculator.solve",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.ADDED,
                        line_start=4,
                        line_end=5,
                        changed_lines=[4, 5],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n"},
            target_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 42\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.Calculator.solve")
        self.assertTrue(any("references added symbol" in r for r in impact.reasons))

    def test_stage2_15_deleted_method_verification(self):
        """Test calls deleted method; verified in base AST and absent from target AST -> SYMBOL, CONFIDENT."""
        test_code = """
from src.calc import Calculator

def test_deleted():
    c = Calculator()
    c.solve()
"""
        test = self._make_test("tests/test_calc.py::test_deleted", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="Calculator.solve",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.DELETED,
                        line_start=4,
                        line_end=5,
                        changed_lines=[4, 5],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n    def solve(self):\n        return 42\n"},
            target_files={"src/calc.py": "class Calculator:\n    def clear(self):\n        pass\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.Calculator.solve")
        self.assertTrue(any("references deleted symbol" in r for r in impact.reasons))

    def test_stage2_16_backward_compatibility_and_deterministic_sorting(self):
        """7 positional fields unpacked correctly, Stage 1 constructor calls work, and outputs sorted by (test_file, test_name, node_id)."""
        # Positional Stage 1 constructor and unpacking
        impact = TestImpact("node1", "test_file.py", "test_foo", ImpactType.MODULE, "mod.py", "mod", ["reason1"])
        node_id, test_file, test_name, impact_type, matched_file, matched_module, reasons = impact
        self.assertEqual(node_id, "node1")
        self.assertEqual(test_file, "test_file.py")
        self.assertEqual(test_name, "test_foo")
        self.assertEqual(impact_type, ImpactType.MODULE)
        self.assertEqual(matched_file, "mod.py")
        self.assertEqual(matched_module, "mod")
        self.assertEqual(reasons, ["reason1"])
        self.assertIsNone(impact.matched_symbol)
        self.assertIsNone(impact.confidence)

        # Stage 2 optional fields
        impact2 = TestImpact(
            "node2",
            "file2.py",
            "test2",
            ImpactType.SYMBOL,
            "mod2.py",
            "mod2",
            ["r2"],
            matched_symbol="mod2.fn",
            confidence=ImpactConfidence.CONFIDENT,
        )
        self.assertEqual(impact2.matched_symbol, "mod2.fn")
        self.assertEqual(impact2.confidence, ImpactConfidence.CONFIDENT)

        # Deterministic sorting check
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]
        t1 = self._make_test("b_dir/test_b.py::test_z", ["src.calc"])
        t2 = self._make_test("a_dir/test_a.py::test_b", ["src.calc"])
        t3 = self._make_test("a_dir/test_a.py::test_a", ["src.calc"])
        discovery = self._make_discovery_result([t1, t2, t3])

        res = analyze_test_impact(symbol_changes, discovery)
        self.assertEqual(len(res.affected_tests), 3)
        self.assertEqual(
            [t.node_id for t in res.affected_tests],
            ["a_dir/test_a.py::test_a", "a_dir/test_a.py::test_b", "b_dir/test_b.py::test_z"],
        )

    def test_stage2_17_added_file_symbol_verification(self):
        """Test referencing a symbol in an entirely added Python file -> SYMBOL, CONFIDENT."""
        test_code = """
from src.new_math import add

def test_add():
    assert add(1, 2) == 3
"""
        test = self._make_test("tests/test_math.py::test_add", ["src.new_math"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/new_math.py",
                old_path=None,
                new_path="src/new_math.py",
                change_type=FileChangeType.ADDED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.ADDED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[1, 2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={},
            target_files={"src/new_math.py": "def add(a, b):\n    return a + b\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_math.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.new_math.add")
        self.assertEqual(impact.matched_file, "src/new_math.py")

    def test_stage2_18_deleted_file_symbol_verification(self):
        """Test referencing a symbol in an entirely deleted Python file -> SYMBOL, CONFIDENT."""
        test_code = """
from src.old_math import old_func

def test_old():
    old_func()
"""
        test = self._make_test("tests/test_math.py::test_old", ["src.old_math"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/old_math.py",
                old_path="src/old_math.py",
                new_path=None,
                change_type=FileChangeType.DELETED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="old_func",
                        fully_qualified_name="old_func",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.DELETED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[1, 2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/old_math.py": "def old_func():\n    pass\n"},
            target_files={},
        )
        test_prov = self._make_test_source_provider({"tests/test_math.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.old_math.old_func")

    def test_stage2_19_unverified_dynamic_attribute_fallback(self):
        """Test calls calc.unknown_dynamic_method() where attribute cannot be verified in AST -> MODULE, CONSERVATIVE."""
        test_code = """
import src.calc as calc

def test_dynamic_attr():
    calc.unknown_dynamic_method()
"""
        test = self._make_test("tests/test_calc.py::test_dynamic_attr", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def add(a, b):\n    return a + b\n"},
            target_files={"src/calc.py": "def add(a, b):\n    return a + b + 0\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("cannot be verified in AST" in r for r in impact.reasons))

    def test_stage2_20_deterministic_ordering_multiple_matched_symbols(self):
        """Test referencing multiple changed symbols has stable, sorted matched_symbol and reasons."""
        test_code = """
from src.calc import multiply, add

def test_both():
    assert add(1, 2) == 3
    assert multiply(2, 3) == 6
"""
        test = self._make_test("tests/test_calc.py::test_both", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    ),
                    ChangedSymbol(
                        symbol_name="multiply",
                        fully_qualified_name="multiply",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=3,
                        line_end=4,
                        changed_lines=[4],
                    ),
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def add(a, b):\n    return a + b\ndef multiply(a, b):\n    return a * b\n"},
            target_files={"src/calc.py": "def add(a, b):\n    return a + b\ndef multiply(a, b):\n    return a * b * 1\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Lexicographically first: 'src.calc.add' before 'src.calc.multiply'
        self.assertEqual(impact.matched_symbol, "src.calc.add")
        self.assertEqual(len(impact.reasons), 2)
        self.assertEqual(
            impact.reasons,
            sorted(impact.reasons),
        )

    def test_stage2_21_bare_relative_package_import(self):
        """Test with 'from . import calculator' matches changed tests.unit.calculator submodule without false positives."""
        test_code = """
from . import calculator

def test_calc_submod():
    calculator.compute()
"""
        test = self._make_test("tests/unit/test_calc.py::test_calc_submod", [])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="tests/unit/calculator.py",
                old_path="tests/unit/calculator.py",
                new_path="tests/unit/calculator.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[],
            )
        ]

        test_prov = self._make_test_source_provider({"tests/unit/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "tests.unit.calculator")

    def test_stage2_22_modified_symbol_revision_mismatch_fallback(self):
        """A symbol marked MODIFIED that cannot be verified in base revision triggers conservative fallback, not UNAFFECTED."""
        test_code = """
from src.calc import solve

def test_solve():
    solve()
"""
        test = self._make_test("tests/test_calc.py::test_solve", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="solve",
                        fully_qualified_name="solve",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        # Base file is empty (solve is absent in base AST, causing structural mismatch for MODIFIED)
        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "# no solve here\n"},
            target_files={"src/calc.py": "def solve():\n    return 42\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("could not be symmetrically verified" in r for r in impact.reasons))

    def test_stage2_23_unpacked_assignment_targets(self):
        """Unpacked tuple and list assignments at module and class levels are found by verify_symbol_in_source."""
        source = """
a, b = 1, 2
[c, d] = [3, 4]

class Config:
    (x, y) = (10, 20)
    [p, (q, r)] = [30, (40, 50)]
"""
        for sym in ["a", "b", "c", "d"]:
            status, node = verify_symbol_in_source(source, sym)
            self.assertEqual(status, VerificationStatus.FOUND, f"Failed for symbol '{sym}'")
            self.assertIsNotNone(node)

        for sym in ["Config.x", "Config.y", "Config.p", "Config.q", "Config.r"]:
            status, node = verify_symbol_in_source(source, sym)
            self.assertEqual(status, VerificationStatus.FOUND, f"Failed for symbol '{sym}'")
            self.assertIsNotNone(node)

    def test_stage2_24_inherited_method_scenario(self):
        """Test calls child.inherited_method() where method is on parent -> conservative fallback."""
        test_code = """
from src.calc import Child

def test_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_calc.py::test_inherited", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        # In target source, Child does not define inherited_method in its AST body
        source_prov = self._make_source_provider(
            base_files={
                "src/calc.py": "class Parent:\n    def inherited_method(self):\n        return 1\nclass Child(Parent):\n    pass\n"
            },
            target_files={
                "src/calc.py": "class Parent:\n    def inherited_method(self):\n        return 2\nclass Child(Parent):\n    pass\n"
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Because Child.inherited_method cannot be verified as a static method in Child AST,
        # it conservatively flags the test as affected!
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("cannot be verified in AST" in r for r in impact.reasons))

    def test_stage2_25_unavailable_source_modified_file_added_symbol_fallback(self):
        """Modified file with ADDED symbol but missing base source must NOT be CONFIDENT; triggers conservative fallback."""
        test_code = """
from src.calc import new_func

def test_new():
    new_func()
"""
        test = self._make_test("tests/test_calc.py::test_new", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="new_func",
                        fully_qualified_name="new_func",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.ADDED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[1, 2],
                    )
                ],
            )
        ]

        # Base source unexpectedly unavailable (None) on a MODIFIED file
        source_prov = self._make_source_provider(
            base_files={},  # missing base_src for src/calc.py!
            target_files={"src/calc.py": "def new_func():\n    pass\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Must NOT be confidently verified as SYMBOL because base revision was unavailable
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("Missing revision AST" in r for r in impact.reasons))

    def test_stage2_26_unavailable_source_modified_file_deleted_symbol_fallback(self):
        """Modified file with DELETED symbol but missing target source must NOT be CONFIDENT; triggers conservative fallback."""
        test_code = """
from src.calc import old_func

def test_old():
    old_func()
"""
        test = self._make_test("tests/test_calc.py::test_old", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="old_func",
                        fully_qualified_name="old_func",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.DELETED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[1, 2],
                    )
                ],
            )
        ]

        # Target source unexpectedly unavailable (None) on a MODIFIED file
        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def old_func():\n    pass\n"},
            target_files={},  # missing target_src for src/calc.py!
        )
        test_prov = self._make_test_source_provider({"tests/test_calc.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Must NOT be confidently verified as SYMBOL because target revision was unavailable
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertTrue(any("Missing revision AST" in r for r in impact.reasons))

    def test_stage2_27_starred_assignment_targets(self):
        """Verify _target_contains_name and verify_symbol_in_source on starred and nested unpacking targets."""
        source = """
first, *remaining = values
head, (*middle, tail) = nested_values

class Container:
    a, *b = items
    x, (*y, z) = complex_items
"""
        for sym in ["first", "remaining", "head", "middle", "tail"]:
            status, node = verify_symbol_in_source(source, sym)
            self.assertEqual(status, VerificationStatus.FOUND, f"Failed for symbol '{sym}'")
            self.assertIsNotNone(node)

        for sym in ["Container.a", "Container.b", "Container.x", "Container.y", "Container.z"]:
            status, node = verify_symbol_in_source(source, sym)
            self.assertEqual(status, VerificationStatus.FOUND, f"Failed for symbol '{sym}'")
            self.assertIsNotNone(node)

    def test_stage2_28_deterministic_fallback_selection_across_input_orders(self):
        """Supplying changes in different input orders produces identical results across all fields."""
        test_code_1 = "import src.a as a\ndef test_1():\n    a.unverified_attr()\n"
        test_code_2 = "import src.b as b\ndef test_2():\n    b.unverified_attr()\n"

        test1 = self._make_test("tests/test_a.py::test_1", ["src.a"])
        test2 = self._make_test("tests/test_b.py::test_2", ["src.b"])
        discovery = self._make_discovery_result([test1, test2])

        chg_a = FileSymbolChanges(
            file_path="src/a.py",
            old_path="src/a.py",
            new_path="src/a.py",
            change_type=FileChangeType.MODIFIED,
            changed_symbols=[
                ChangedSymbol(
                    symbol_name="dummy_a",
                    fully_qualified_name="dummy_a",
                    symbol_type=SymbolType.FUNCTION,
                    change_category=ChangeCategory.MODIFIED,
                    line_start=1,
                    line_end=2,
                    changed_lines=[1],
                )
            ],
        )
        chg_b = FileSymbolChanges(
            file_path="src/b.py",
            old_path="src/b.py",
            new_path="src/b.py",
            change_type=FileChangeType.MODIFIED,
            changed_symbols=[
                ChangedSymbol(
                    symbol_name="dummy_b",
                    fully_qualified_name="dummy_b",
                    symbol_type=SymbolType.FUNCTION,
                    change_category=ChangeCategory.MODIFIED,
                    line_start=1,
                    line_end=2,
                    changed_lines=[1],
                )
            ],
        )

        source_prov = self._make_source_provider(
            base_files={"src/a.py": "def dummy_a(): pass\n", "src/b.py": "def dummy_b(): pass\n"},
            target_files={"src/a.py": "def dummy_a(): pass\n", "src/b.py": "def dummy_b(): pass\n"},
        )
        test_prov = self._make_test_source_provider({
            "tests/test_a.py": test_code_1,
            "tests/test_b.py": test_code_2,
        })

        # Order 1: [chg_a, chg_b]
        res1 = analyze_test_impact(
            [chg_a, chg_b],
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        # Order 2: [chg_b, chg_a]
        res2 = analyze_test_impact(
            [chg_b, chg_a],
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res1.total_discovered_tests, res2.total_discovered_tests)
        self.assertEqual(res1.total_affected_tests, res2.total_affected_tests)
        self.assertEqual(res1.unaffected_tests, res2.unaffected_tests)
        self.assertEqual(len(res1.affected_tests), len(res2.affected_tests))

        for t1, t2 in zip(res1.affected_tests, res2.affected_tests):
            self.assertEqual(t1.node_id, t2.node_id)
            self.assertEqual(t1.test_file, t2.test_file)
            self.assertEqual(t1.test_name, t2.test_name)
            self.assertEqual(t1.impact_type, t2.impact_type)
            self.assertEqual(t1.matched_file, t2.matched_file)
            self.assertEqual(t1.matched_module, t2.matched_module)
            self.assertEqual(t1.matched_symbol, t2.matched_symbol)
            self.assertEqual(t1.confidence, t2.confidence)
            self.assertEqual(t1.reasons, t2.reasons)

    def test_stage2_29_cross_file_inheritance_scenario(self):
        """Cross-file inheritance: src/child.py extends Parent from src/base.py; modifying Parent triggers conservative impact."""
        test_code = """
from src.child import Child

def test_cross_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_child.py::test_cross_inherited", ["src.child"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        # Only src/base.py is changed. src/child.py imports Parent from src.base
        source_prov = self._make_source_provider(
            base_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 1\n",
                "src/child.py": "from src.base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
            target_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 2\n",
                "src/child.py": "from src.base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_child.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Correctly identifies conservative module impact on src.base
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.base")
        self.assertEqual(impact.matched_file, "src/base.py")
        self.assertTrue(any("cross-file inheritance or dependency" in r for r in impact.reasons))

    def test_stage2_30_cross_file_inheritance_direct_relative_import(self):
        """Cross-file inheritance using direct relative import: src/child.py uses 'from .base import Parent'."""
        test_code = """
from src.child import Child

def test_relative_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_child.py::test_relative_inherited", ["src.child"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 1\n",
                "src/child.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
            target_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 2\n",
                "src/child.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_child.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.base")
        self.assertEqual(impact.matched_file, "src/base.py")
        self.assertTrue(any("cross-file inheritance or dependency" in r for r in impact.reasons))

    def test_stage2_31_cross_file_inheritance_nested_relative_import(self):
        """Cross-file inheritance using nested relative import: src/nested/child.py uses 'from ..base import Parent'."""
        test_code = """
from src.nested.child import Child

def test_nested_relative_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_nested_child.py::test_nested_relative_inherited", ["src.nested.child"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 1\n",
                "src/nested/child.py": "from ..base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
            target_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 2\n",
                "src/nested/child.py": "from ..base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_nested_child.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.base")
        self.assertEqual(impact.matched_file, "src/base.py")
        self.assertTrue(any("cross-file inheritance or dependency" in r for r in impact.reasons))

    def test_stage2_32_cross_file_inheritance_nested_init_py_relative_import(self):
        """Package __init__.py relative import: src/nested/__init__.py imports from .base -> src.nested.base."""
        test_code = """
from src.nested import Child

def test_nested_init_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_nested_init.py::test_nested_init_inherited", ["src.nested"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/nested/base.py",
                old_path="src/nested/base.py",
                new_path="src/nested/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={
                "src/nested/base.py": "class Parent:\n    def inherited_method(self):\n        return 1\n",
                "src/nested/__init__.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
            target_files={
                "src/nested/base.py": "class Parent:\n    def inherited_method(self):\n        return 2\n",
                "src/nested/__init__.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_nested_init.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.nested.base")
        self.assertEqual(impact.matched_file, "src/nested/base.py")
        self.assertTrue(any("cross-file inheritance or dependency" in r for r in impact.reasons))

    def test_stage2_33_cross_file_inheritance_root_init_py_relative_import(self):
        """Root package __init__.py relative import: src/__init__.py imports from .base -> src.base."""
        test_code = """
from src import Child

def test_root_init_inherited():
    c = Child()
    c.inherited_method()
"""
        test = self._make_test("tests/test_root_init.py::test_root_init_inherited", ["src"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="inherited_method",
                        fully_qualified_name="Parent.inherited_method",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 1\n",
                "src/__init__.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
            target_files={
                "src/base.py": "class Parent:\n    def inherited_method(self):\n        return 2\n",
                "src/__init__.py": "from .base import Parent\n\nclass Child(Parent):\n    pass\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_root_init.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.base")
        self.assertEqual(impact.matched_file, "src/base.py")
        self.assertTrue(any("cross-file inheritance or dependency" in r for r in impact.reasons))

    def test_stage2_34_source_and_ast_caching(self):
        """Finding 1: Repeated revision source and AST verifications must use the cache."""
        test_code_1 = """
from src.calc import add

def test_add_1():
    add(1, 2)
"""
        test_code_2 = """
from src.calc import add

def test_add_2():
    add(3, 4)
"""
        test1 = self._make_test("tests/test_calc1.py::test_add_1", ["src.calc"])
        test2 = self._make_test("tests/test_calc2.py::test_add_2", ["src.calc"])
        discovery = self._make_discovery_result([test1, test2])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        calls = {"base": 0, "target": 0}

        def counting_source_provider(rev_label: str, file_path: str) -> Optional[str]:
            calls[rev_label] += 1
            if rev_label == "base":
                return "def add(a, b):\n    return a + b\n"
            return "def add(a, b):\n    return a + b + 1\n"

        test_prov = self._make_test_source_provider({
            "tests/test_calc1.py": test_code_1,
            "tests/test_calc2.py": test_code_2,
        })

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=counting_source_provider,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 2)
        # Even though there are 2 tests verifying 'add' in 'src/calc.py', source_provider
        # must be called exactly ONCE per revision due to caching!
        self.assertEqual(calls["base"], 1)
        self.assertEqual(calls["target"], 1)

    def test_stage2_35_unavailable_relevant_module_source_fallback(self):
        """Finding 4: When a relevant module's source is unavailable, apply conservative fallback."""
        test_code = """
from src.unresolvable_child import SpecialChild

def test_unresolvable():
    c = SpecialChild()
    c.execute_action()
"""
        test = self._make_test("tests/test_unresolvable.py::test_unresolvable", ["src.unresolvable_child"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="execute_action",
                        fully_qualified_name="Parent.execute_action",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            )
        ]

        # 'src/unresolvable_child.py' is NOT in base_files or target_files (source unavailable)
        source_prov = self._make_source_provider(
            base_files={
                "src/base.py": "class Parent:\n    def execute_action(self):\n        return 1\n"
            },
            target_files={
                "src/base.py": "class Parent:\n    def execute_action(self):\n        return 2\n"
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_unresolvable.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        self.assertEqual(impact.impact_type, ImpactType.MODULE)
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)
        self.assertEqual(impact.matched_module, "src.base")
        self.assertTrue(any("source is unavailable while related module 'src.base' changed" in r for r in impact.reasons))

    def test_stage2_36_unavailable_unrelated_module_source_ignored(self):
        """Finding 4: Unrelated external module whose source is unavailable must NOT mark tests affected."""
        test_code = """
import external_lib

def test_external():
    external_lib.helper_func()
"""
        test = self._make_test("tests/test_external.py::test_external", ["external_lib"])
        discovery = self._make_discovery_result([test])

        # Repo change is in src/calculator.py
        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calculator.py",
                old_path="src/calculator.py",
                new_path="src/calculator.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calculator.py": "def add(a, b): return a + b\n"},
            target_files={"src/calculator.py": "def add(a, b): return a + b + 1\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_external.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        # Unrelated test must remain UNAFFECTED
        self.assertEqual(res.total_affected_tests, 0)
        self.assertIn("tests/test_external.py::test_external", res.unaffected_tests)

    def test_stage2_37_ordinary_expressions_do_not_trigger_module_impact(self):
        """Finding 3: Test with f-strings and arithmetic still gets precise SYMBOL impact, not conservative MODULE."""
        test_code = """
from src.calc import add

def test_calc_expr():
    prefix = f"Prefix_{10}"
    clean = prefix.strip()
    val = 1 + 2
    res = add(val, 4)
    assert res == 7 and clean
"""
        test = self._make_test("tests/test_expr.py::test_calc_expr", ["src.calc"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def add(a, b): return a + b\n"},
            target_files={"src/calc.py": "def add(a, b): return a + b + 1\n"},
        )
        test_prov = self._make_test_source_provider({"tests/test_expr.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Must retain precise SYMBOL impact, NOT degrade to MODULE impact!
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)
        self.assertEqual(impact.matched_symbol, "src.calc.add")

    def test_stage2_38_ast_parsing_not_repeated(self):
        """Finding 1 / Requirement 2: Verify directly that ast.parse is called once per revision/file."""
        import ast
        from unittest.mock import patch

        test_code_1 = "from src.calc import add\ndef test_1(): add(1, 2)\n"
        test_code_2 = "from src.calc import add\ndef test_2(): add(3, 4)\n"
        test_code_3 = "from src.calc import add\ndef test_3(): add(5, 6)\n"

        test1 = self._make_test("tests/test1.py::test_1", ["src.calc"])
        test2 = self._make_test("tests/test2.py::test_2", ["src.calc"])
        test3 = self._make_test("tests/test3.py::test_3", ["src.calc"])
        discovery = self._make_discovery_result([test1, test2, test3])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/calc.py",
                old_path="src/calc.py",
                new_path="src/calc.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/calc.py": "def add(a, b): return a + b\n"},
            target_files={"src/calc.py": "def add(a, b): return a + b + 1\n"},
        )
        test_prov = self._make_test_source_provider({
            "tests/test1.py": test_code_1,
            "tests/test2.py": test_code_2,
            "tests/test3.py": test_code_3,
        })

        parsed_sources: list[str] = []
        original_ast_parse = ast.parse

        def tracking_ast_parse(source, *args, **kwargs):
            parsed_sources.append(source)
            return original_ast_parse(source, *args, **kwargs)

        with patch("ast.parse", side_effect=tracking_ast_parse):
            res = analyze_test_impact(
                symbol_changes,
                discovery,
                source_provider=source_prov,
                test_source_provider=test_prov,
            )

        self.assertEqual(res.total_affected_tests, 3)
        # Check how many times src/calc.py content was parsed across the 3 tests:
        # Base rev was parsed exactly ONCE
        base_parses = [s for s in parsed_sources if "return a + b\n" in s]
        self.assertEqual(len(base_parses), 1)
        # Target rev was parsed exactly ONCE
        target_parses = [s for s in parsed_sources if "return a + b + 1\n" in s]
        self.assertEqual(len(target_parses), 1)

    def test_stage2_39_empty_source_vs_missing_source_cache_distinction(self):
        """Finding 1 / Requirement 2: Empty source and missing source are cached distinctly."""
        base_empty = ""
        target_src = "def add(a, b): return a + b\n"

        test_code = "from src.empty import add\ndef test_add(): add(1, 2)\n"
        test = self._make_test("tests/test_empty.py::test_add", ["src.empty"])
        discovery = self._make_discovery_result([test])

        symbol_changes = [
            FileSymbolChanges(
                file_path="src/empty.py",
                old_path="src/empty.py",
                new_path="src/empty.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="add",
                        fully_qualified_name="add",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.ADDED,
                        line_start=1,
                        line_end=1,
                        changed_lines=[1],
                    )
                ],
            )
        ]

        source_prov = self._make_source_provider(
            base_files={"src/empty.py": base_empty},
            target_files={"src/empty.py": target_src},
        )
        test_prov = self._make_test_source_provider({"tests/test_empty.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # In base_empty, 'add' is confirmed absent in parsed empty AST (not missing source)
        # In target_src, 'add' is found. So change_category.ADDED is matched CONFIDENTLY!
        self.assertEqual(impact.impact_type, ImpactType.SYMBOL)
        self.assertEqual(impact.confidence, ImpactConfidence.CONFIDENT)

    def test_stage2_40_relevant_module_selection_when_multiple_packages_change(self):
        """Finding 4 / Requirement 3: When multiple packages change, select genuinely related package."""
        test_code = """
from src.unresolvable_child import SpecialChild

def test_unresolvable():
    c = SpecialChild()
    c.execute_action()
"""
        test = self._make_test("tests/test_unresolvable.py::test_unresolvable", ["src.unresolvable_child"])
        discovery = self._make_discovery_result([test])

        # Two packages changed: alpha_pkg and src
        # Alphabetically, 'alpha_pkg' comes first. But 'src.unresolvable_child' belongs to 'src',
        # so it must select 'src.base' as the genuinely related module!
        symbol_changes = [
            FileSymbolChanges(
                file_path="alpha_pkg/worker.py",
                old_path="alpha_pkg/worker.py",
                new_path="alpha_pkg/worker.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="work",
                        fully_qualified_name="work",
                        symbol_type=SymbolType.FUNCTION,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=1,
                        line_end=2,
                        changed_lines=[2],
                    )
                ],
            ),
            FileSymbolChanges(
                file_path="src/base.py",
                old_path="src/base.py",
                new_path="src/base.py",
                change_type=FileChangeType.MODIFIED,
                changed_symbols=[
                    ChangedSymbol(
                        symbol_name="execute_action",
                        fully_qualified_name="Parent.execute_action",
                        symbol_type=SymbolType.METHOD,
                        change_category=ChangeCategory.MODIFIED,
                        line_start=2,
                        line_end=3,
                        changed_lines=[3],
                    )
                ],
            ),
        ]

        source_prov = self._make_source_provider(
            base_files={
                "alpha_pkg/worker.py": "def work(): return 1\n",
                "src/base.py": "class Parent:\n    def execute_action(self):\n        return 1\n",
            },
            target_files={
                "alpha_pkg/worker.py": "def work(): return 2\n",
                "src/base.py": "class Parent:\n    def execute_action(self):\n        return 2\n",
            },
        )
        test_prov = self._make_test_source_provider({"tests/test_unresolvable.py": test_code})

        res = analyze_test_impact(
            symbol_changes,
            discovery,
            source_provider=source_prov,
            test_source_provider=test_prov,
        )

        self.assertEqual(res.total_affected_tests, 1)
        impact = res.affected_tests[0]
        # Must select 'src.base' because it shares the root package 'src', NOT 'alpha_pkg.worker'!
        self.assertEqual(impact.matched_module, "src.base")
        self.assertEqual(impact.matched_file, "src/base.py")
        self.assertEqual(impact.confidence, ImpactConfidence.CONSERVATIVE)


if __name__ == "__main__":
    unittest.main()
