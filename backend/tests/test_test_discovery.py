"""Unit tests for the Test Discovery service."""

import tempfile
import unittest
from pathlib import Path

from services.test_discovery import (
    DiscoveredTest,
    DiscoveryResult,
    TestDiscoveryError,
    discover_tests,
    discover_tests_in_file,
)


class TestTestDiscovery(unittest.TestCase):
    """Comprehensive test suite for test_discovery module."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_function_and_class_tests(self):
        """Verify discovery of standalone functions and standard pytest class methods."""
        test_file = self.root_path / "test_sample.py"
        test_file.write_text(
            """
import math
from os import path

def test_standalone():
    assert 1 + 1 == 2

async def test_async_function():
    assert True

def helper_function():
    return 42

class TestCalculator:
    def test_add(self):
        assert 1 + 1 == 2

    def not_a_test(self):
        pass

class TestIgnoredDueToInit:
    def __init__(self):
        pass

    def test_should_be_ignored(self):
        assert False

class RegularClass:
    def test_ignored_in_non_test_class(self):
        pass
""",
            encoding="utf-8",
        )

        results = discover_tests_in_file(test_file, self.root_path)
        test_names = [t.test_name for t in results]

        self.assertIn("test_standalone", test_names)
        self.assertIn("test_async_function", test_names)
        self.assertIn("TestCalculator::test_add", test_names)
        self.assertNotIn("helper_function", test_names)
        self.assertNotIn("TestCalculator::not_a_test", test_names)
        self.assertNotIn("TestIgnoredDueToInit::test_should_be_ignored", test_names)
        self.assertNotIn("RegularClass::test_ignored_in_non_test_class", test_names)

        # Check imported modules extraction
        self.assertIn("math", results[0].imported_modules)
        self.assertIn("os.path", results[0].imported_modules)

    def test_discover_tests_across_directory(self):
        """Verify recursive discovery across nested directories and ignored dirs."""
        tests_dir = self.root_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").write_text("def test_one(): pass", encoding="utf-8")

        nested_dir = tests_dir / "unit"
        nested_dir.mkdir()
        (nested_dir / "test_b.py").write_text("def test_two(): pass", encoding="utf-8")

        # Files in ignored dirs should not be discovered
        ignored_dir = self.root_path / ".venv" / "tests"
        ignored_dir.mkdir(parents=True)
        (ignored_dir / "test_ignored.py").write_text("def test_three(): pass", encoding="utf-8")

        result = discover_tests(self.root_path)
        self.assertEqual(result.total_tests, 2)
        self.assertEqual(result.test_files_count, 2)
        discovered_node_ids = [t.node_id for t in result.tests]
        self.assertIn("tests/test_a.py::test_one", discovered_node_ids)
        self.assertIn("tests/unit/test_b.py::test_two", discovered_node_ids)

    def test_invalid_repository_path(self):
        """Verify error raised for non-existent path."""
        with self.assertRaises(TestDiscoveryError):
            discover_tests(self.root_path / "does_not_exist")

    def test_syntax_error_file_handled_gracefully(self):
        """Verify files with syntax errors do not crash the discovery."""
        test_file = self.root_path / "test_syntax_error.py"
        test_file.write_text("def invalid syntax::: pass", encoding="utf-8")

        results = discover_tests_in_file(test_file, self.root_path)
        self.assertEqual(results, [])

    def test_empty_test_file(self):
        """Verify empty (0-byte) test files return no tests without errors."""
        empty_file = self.root_path / "test_empty.py"
        empty_file.write_text("", encoding="utf-8")

        results = discover_tests_in_file(empty_file, self.root_path)
        self.assertEqual(results, [])

    def test_relative_imports(self):
        """Verify relative imports are extracted accurately with relative prefixes."""
        test_file = self.root_path / "test_rel.py"
        test_file.write_text(
            """
from . import helper
from .calculator import add
from ..module import function

def test_rel():
    pass
""",
            encoding="utf-8",
        )

        results = discover_tests_in_file(test_file, self.root_path)
        self.assertEqual(len(results), 1)
        imports = results[0].imported_modules

        # Verify from . import helper
        self.assertIn(".helper", imports)
        self.assertIn("helper", imports)

        # Verify from .calculator import add
        self.assertIn(".calculator", imports)
        self.assertIn(".calculator.add", imports)
        self.assertIn("add", imports)

        # Verify from ..module import function
        self.assertIn("..module", imports)
        self.assertIn("..module.function", imports)
        self.assertIn("function", imports)

    def test_utf8_and_non_ascii_characters(self):
        """Verify test files with UTF-8 non-ASCII characters parse cleanly."""
        test_file = self.root_path / "test_unicode.py"
        test_file.write_text(
            """
# Überprüfen von mathematischen Funktionen 🚀
def test_unicode_strings():
    greeting = "Bonjour le monde! 🌍"
    assert "Bonjour" in greeting

def test_accented_identifiers():
    val = "crème brûlée"
    assert len(val) > 0
""",
            encoding="utf-8",
        )

        results = discover_tests_in_file(test_file, self.root_path)
        self.assertEqual(len(results), 2)
        names = [t.test_name for t in results]
        self.assertIn("test_unicode_strings", names)
        self.assertIn("test_accented_identifiers", names)

    def test_repository_containing_no_tests(self):
        """Verify repository with source files but no test files returns zero tests."""
        src_dir = self.root_path / "src"
        src_dir.mkdir()
        (src_dir / "calculator.py").write_text("def add(a, b): return a + b", encoding="utf-8")
        (src_dir / "utils.py").write_text("def log(): pass", encoding="utf-8")

        result = discover_tests(self.root_path)
        self.assertEqual(result.total_tests, 0)
        self.assertEqual(result.test_files_count, 0)
        self.assertEqual(result.tests, [])

    def test_class_containing_helper_methods(self):
        """Verify non-test helper methods and setup methods inside test classes are excluded."""
        test_file = self.root_path / "test_helpers.py"
        test_file.write_text(
            """
class TestWithHelpers:
    def setUp(self):
        self.value = 10

    def tearDown(self):
        self.value = 0

    def _private_helper(self):
        return self.value * 2

    def calculate_fixture(self):
        return 42

    def test_actual_computation(self):
        assert self._private_helper() == 20
""",
            encoding="utf-8",
        )

        results = discover_tests_in_file(test_file, self.root_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].test_name, "TestWithHelpers::test_actual_computation")
        self.assertEqual(results[0].node_id, "test_helpers.py::TestWithHelpers::test_actual_computation")

    def test_unittest_testcase_subclasses_without_test_prefix(self):
        """Verify unittest.TestCase subclasses without 'Test' prefix are discovered."""
        test_file = self.root_path / "test_suite.py"
        test_file.write_text(
            """
import unittest
from unittest import TestCase

class CalculatorSuite(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 3, 5)

    def helper_method(self):
        pass

class CustomValidationSuite(TestCase):
    def test_subtraction(self):
        self.assertEqual(5 - 2, 3)
""",
            encoding="utf-8",
        )

        results = discover_tests_in_file(test_file, self.root_path)
        test_names = [t.test_name for t in results]
        node_ids = [t.node_id for t in results]

        self.assertIn("CalculatorSuite::test_addition", test_names)
        self.assertIn("CustomValidationSuite::test_subtraction", test_names)
        self.assertNotIn("CalculatorSuite::helper_method", test_names)

        self.assertIn("test_suite.py::CalculatorSuite::test_addition", node_ids)
        self.assertIn("test_suite.py::CustomValidationSuite::test_subtraction", node_ids)


if __name__ == "__main__":
    unittest.main()
