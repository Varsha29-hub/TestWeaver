"""Unit tests for TestSymbolExtractor (Week 3, Phase 3 — Stage 2).

Verifies AST-based test function symbol dependency extraction, including:
- Multiple dotted imports sharing a package root.
- Module and function aliases.
- Direct function and class references.
- Class instantiation and instance method resolution.
- Async test functions and methods.
- Wildcard import detection.
- Dynamic reflection detection (including aliased importlib).
- Unresolved instance and attribute detection.
- Per-test reference isolation.
- Function-local import isolation between tests.
- Unresolved factory-returned object detection.
- Preservation of unresolved relative-import dependency info.
- Syntactic attribute resolution vs. unverified symbol tracking.
- Parameter shadowing of imported functions and module aliases.
- Nested function parameter scope isolation.
- Parameter shadowing during class instantiation and constructors.
"""

from __future__ import annotations

import unittest

from services.test_symbol_extractor import TestSymbolExtractor


class TestTestSymbolExtractor(unittest.TestCase):
    """Test suite for TestSymbolExtractor."""

    def test_multiple_dotted_imports_sharing_package_root(self) -> None:
        """Verify multiple dotted imports sharing root 'src' resolve attribute chains correctly."""
        code = """
import src.calculator
import src.statistics

def test_calculator_add():
    res = src.calculator.add(1, 2)
    assert res == 3

def test_statistics_mean():
    avg = src.statistics.mean([1, 2, 3])
    assert avg == 2
"""
        extractor = TestSymbolExtractor(code, "tests/test_math.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 2)
        calc_dep = next(d for d in deps if d.test_name == "test_calculator_add")
        stats_dep = next(d for d in deps if d.test_name == "test_statistics_mean")

        self.assertIn("src.calculator.add", calc_dep.referenced_symbols)
        self.assertNotIn("src.statistics.mean", calc_dep.referenced_symbols)
        self.assertFalse(calc_dep.has_uncertainty)

        self.assertIn("src.statistics.mean", stats_dep.referenced_symbols)
        self.assertNotIn("src.calculator.add", stats_dep.referenced_symbols)
        self.assertFalse(stats_dep.has_uncertainty)

    def test_module_aliases(self) -> None:
        """Verify 'import ... as ...' resolves aliased module calls."""
        code = """
import src.calculator as calc

def test_aliased_calc():
    assert calc.add(1, 2) == 3
    assert calc.Calculator.solve() == 42
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)
        self.assertIn("src.calculator.Calculator.solve", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_function_and_symbol_aliases(self) -> None:
        """Verify 'from ... import ... as ...' resolves function aliases."""
        code = """
from src.calculator import add as plus, subtract as minus

def test_function_aliases():
    assert plus(1, 2) == 3
    assert minus(5, 2) == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)
        self.assertIn("src.calculator.subtract", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_direct_function_and_class_references(self) -> None:
        """Verify direct imported names and class attribute lookups."""
        code = """
from src.calculator import add, Calculator

def test_direct_refs():
    assert add(10, 20) == 30
    val = Calculator.DEFAULT_TIMEOUT
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)
        self.assertIn("src.calculator.Calculator.DEFAULT_TIMEOUT", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_class_instantiation_and_instance_method_resolution(self) -> None:
        """Verify 'calculator = Calculator(); calculator.solve()' resolves to Class.method."""
        code = """
from src.calc import Calculator

def test_instance_resolution():
    calculator = Calculator()
    res = calculator.solve()
    assert res == 100
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calc.Calculator.solve", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_class_instantiation_via_module_alias_and_package_root(self) -> None:
        """Verify instance method resolution when class is instantiated via alias or package root."""
        code = """
import src.calc as calc
import src.processor

def test_alias_and_root_instantiation():
    c1 = calc.Calculator()
    c1.compute()

    p1 = src.processor.Processor()
    p1.run()
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calc.Calculator.compute", dep.referenced_symbols)
        self.assertIn("src.processor.Processor.run", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_unittest_class_with_setup_self_instances(self) -> None:
        """Verify TestCase setUp assigning self.calc resolves in test methods."""
        code = """
from src.calc import Calculator

class TestCalculatorService:
    def setUp(self):
        self.calc = Calculator()

    def test_method_solve(self):
        assert self.calc.solve() == 42
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertEqual(
            dep.node_id,
            "tests/test_calc.py::TestCalculatorService::test_method_solve",
        )
        self.assertIn("src.calc.Calculator.solve", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_async_test_functions(self) -> None:
        """Verify async test functions and awaited instance method calls."""
        code = """
from src.calc import Calculator

async def test_async_compute():
    calc = Calculator()
    result = await calc.async_solve()
    assert result == 10
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertEqual(dep.test_name, "test_async_compute")
        self.assertIn("src.calc.Calculator.async_solve", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_wildcard_import_detection(self) -> None:
        """Verify wildcard import 'from ... import *' flags uncertainty on all tests."""
        code = """
from src.calculator import *

def test_wildcard_call():
    assert add(1, 2) == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Wildcard import" in reason for reason in dep.uncertainty_reasons)
        )

    def test_dynamic_reflection_detection(self) -> None:
        """Verify dynamic reflection calls like getattr and eval flag uncertainty."""
        code = """
import src.calculator as calc

def test_getattr_dynamic():
    fn = getattr(calc, "add")
    assert fn(1, 2) == 3

def test_eval_dynamic():
    res = eval("calc.add(1, 2)")
    assert res == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 2)
        for dep in deps:
            self.assertTrue(dep.has_uncertainty)
            self.assertTrue(
                any("Dynamic reflection" in r for r in dep.uncertainty_reasons)
            )

    def test_unresolved_instance_method_detection(self) -> None:
        """Verify calling methods on unknown parameters/fixtures flags uncertainty."""
        code = """
import src.calculator

def test_fixture_call(custom_fixture):
    custom_fixture.execute()
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Unresolved instance method call" in r for r in dep.uncertainty_reasons)
        )

    def test_unresolved_package_root_subpath(self) -> None:
        """Verify access to un-imported subpath under imported root flags uncertainty."""
        code = """
import src.calculator

def test_unimported_subpath():
    src.unknown_module.do_something()
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Unresolved subpath" in r for r in dep.uncertainty_reasons)
        )

    def test_per_test_reference_isolation(self) -> None:
        """Verify referenced symbols and local variables are strictly isolated per test."""
        code = """
from src.calculator import add, subtract, multiply

def test_addition():
    assert add(1, 2) == 3

def test_subtraction():
    assert subtract(5, 2) == 3

def test_multiplication():
    assert multiply(2, 4) == 8
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 3)
        add_dep = next(d for d in deps if d.test_name == "test_addition")
        sub_dep = next(d for d in deps if d.test_name == "test_subtraction")
        mul_dep = next(d for d in deps if d.test_name == "test_multiplication")

        self.assertEqual(add_dep.referenced_symbols, {"src.calculator.add"})
        self.assertEqual(sub_dep.referenced_symbols, {"src.calculator.subtract"})
        self.assertEqual(mul_dep.referenced_symbols, {"src.calculator.multiply"})

    def test_builtin_literals_do_not_flag_uncertainty(self) -> None:
        """Verify standard operations on builtins (list, dict, str) do not cause false uncertainty."""
        code = """
from src.calculator import add

def test_with_literals():
    total = add(1, 2)
    items = [total]
    items.append(4)
    text = "result"
    text.upper()
    mapping = {"a": 1}
    mapping.get("a")
    assert len(items) == 2
"""
        extractor = TestSymbolExtractor(code, "tests/test_calc.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)
        self.assertFalse(dep.has_uncertainty)

    def test_local_import_isolation_between_tests(self) -> None:
        """Regression test for Issue 1: imports inside test functions must not leak to other tests."""
        code = """
def test_first():
    from src.calculator import add
    add(1, 2)

def test_second():
    add(3, 4)
"""
        extractor = TestSymbolExtractor(code, "tests/test_leak.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 2)
        first_dep = next(d for d in deps if d.test_name == "test_first")
        second_dep = next(d for d in deps if d.test_name == "test_second")

        self.assertIn("src.calculator.add", first_dep.referenced_symbols)
        # In second test, 'add' is an unbound name and must NOT resolve using first test's import
        self.assertNotIn("src.calculator.add", second_dep.referenced_symbols)

    def test_unresolved_factory_returned_object(self) -> None:
        """Regression test for Issue 2: factory-returned objects and local assignments flag uncertainty."""
        code = """
import src.calculator

def test_service():
    obj = create_service()
    obj.execute()
"""
        extractor = TestSymbolExtractor(code, "tests/test_factory.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Unresolved instance method call on 'obj.execute'" in r for r in dep.uncertainty_reasons)
        )

    def test_unresolved_local_instance_method_calls(self) -> None:
        """Regression test for Issue 2: chained or helper-assigned instances flag uncertainty."""
        code = """
import src.calculator

def test_helper_assigned():
    helper = get_helper(42)
    result = helper.compute()
    assert result is not None
"""
        extractor = TestSymbolExtractor(code, "tests/test_helper.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Unresolved instance method call on 'helper.compute'" in r for r in dep.uncertainty_reasons)
        )

    def test_preserves_unresolved_relative_import_information(self) -> None:
        """Regression test for Issue 3: relative imports preserve unresolved module paths explicitly."""
        code = """
from ..calculator import add

def test_relative_import():
    add(1, 2)
"""
        extractor = TestSymbolExtractor(code, "tests/subpkg/test_relative.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(dep.has_unresolved_relative_imports)
        self.assertEqual(dep.unresolved_relative_imports, ["..calculator"])
        self.assertTrue(
            any("Relative import 'from ..calculator' requires package context" in r for r in dep.uncertainty_reasons)
        )

    def test_dynamic_import_through_module_alias(self) -> None:
        """Regression test for Issue 2: 'import importlib as il; il.import_module()' must be detected."""
        code = """
import importlib as il

def test_dynamic_alias():
    mod = il.import_module("src.calculator")
    assert mod is not None
"""
        extractor = TestSymbolExtractor(code, "tests/test_dynamic_import.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Dynamic import call: 'il.import_module()'" in r for r in dep.uncertainty_reasons)
        )

    def test_dynamic_import_through_function_import(self) -> None:
        """Regression test for Issue 2: 'from importlib import import_module; import_module()' detected."""
        code = """
from importlib import import_module

def test_dynamic_func():
    mod = import_module("src.calculator")
    assert mod is not None
"""
        extractor = TestSymbolExtractor(code, "tests/test_dynamic_import.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Dynamic import call: 'import_module()'" in r for r in dep.uncertainty_reasons)
        )

    def test_dynamic_import_through_aliased_function_import(self) -> None:
        """Regression test for Issue 2: 'from importlib import import_module as imp; imp()' detected."""
        code = """
from importlib import import_module as imp

def test_dynamic_aliased_func():
    mod = imp("src.calculator")
    assert mod is not None
"""
        extractor = TestSymbolExtractor(code, "tests/test_dynamic_import.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Dynamic import call: 'imp()'" in r for r in dep.uncertainty_reasons)
        )

    def test_distinguishes_syntactic_resolution_from_verified_symbols(self) -> None:
        """Regression test for Issue 3: syntactic attribute chains are preserved as unverified references."""
        code = """
import src.calculator as calc

def test_example():
    calc.unknown_method()
"""
        extractor = TestSymbolExtractor(code, "tests/test_unverified.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # Candidate FQN is generated syntactically
        self.assertIn("src.calculator.unknown_method", dep.referenced_symbols)
        # But preserved explicitly as unverified for cross-referencing by the integration layer
        self.assertTrue(dep.has_unverified_symbol_references)
        self.assertIn("src.calculator.unknown_method", dep.unverified_symbol_references)
        self.assertIn("unknown_method", dep.module_attribute_references["src.calculator"])

    def test_instance_method_unverified_attribute_tracking(self) -> None:
        """Regression test for Issue 3: instance method calls track unverified class attribute access."""
        code = """
from src.calc import Calculator

def test_instance_unverified():
    calculator = Calculator()
    calculator.unknown_method()
"""
        extractor = TestSymbolExtractor(code, "tests/test_unverified_instance.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calc.Calculator.unknown_method", dep.referenced_symbols)
        self.assertTrue(dep.has_unverified_symbol_references)
        self.assertIn("src.calc.Calculator.unknown_method", dep.unverified_symbol_references)
        self.assertIn("Calculator.unknown_method", dep.module_attribute_references["src.calc"])

    def test_imported_function_shadowed_by_parameter(self) -> None:
        """Regression test: an imported function shadowed by a test parameter must NOT resolve to imported FQN."""
        code = """
from src.calculator import add

def test_fixture(add):
    add(1, 2)
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)

    def test_imported_module_alias_shadowed_by_parameter(self) -> None:
        """Regression test: an imported module alias shadowed by a parameter must NOT resolve to module symbol."""
        code = """
import src.calculator as calc

def test_fixture(calc):
    calc.execute()
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.execute", dep.referenced_symbols)
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("parameter 'calc.execute'" in r for r in dep.uncertainty_reasons)
        )

    def test_unrelated_function_resolves_when_another_is_shadowed(self) -> None:
        """Regression test: an unrelated imported function still resolves correctly when another is shadowed."""
        code = """
from src.calculator import add, subtract

def test_fixture(add):
    add(1, 2)
    subtract(5, 2)
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)
        self.assertIn("src.calculator.subtract", dep.referenced_symbols)

    def test_normal_module_alias_resolves_when_not_shadowed(self) -> None:
        """Regression test: a normal imported module alias resolves correctly when not shadowed."""
        code = """
import src.calculator as calc

def test_normal():
    calc.execute()
"""
        extractor = TestSymbolExtractor(code, "tests/test_normal.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.execute", dep.referenced_symbols)

    def test_nested_function_parameters_do_not_leak_to_enclosing_scope(self) -> None:
        """Regression test: nested function parameters must not shadow imported symbols in enclosing test scope."""
        code = """
from src.calculator import add

def test_outer():
    def helper(add):
        return add + 1
    res = add(1, 2)
    assert res == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_nested_scope.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)

    def test_imported_class_constructor_shadowed_by_parameter(self) -> None:
        """Verify imported class constructor shadowed by parameter does not resolve to imported class."""
        code = """
from src.calculator import Calculator

def test_fixture(Calculator):
    obj = Calculator()
    obj.solve()
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow_class.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.Calculator", dep.referenced_symbols)
        self.assertNotIn("src.calculator.Calculator.solve", dep.referenced_symbols)
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("Unresolved instance method call on 'obj.solve'" in r for r in dep.uncertainty_reasons)
        )

    def test_module_alias_constructor_shadowed_by_parameter(self) -> None:
        """Verify module alias shadowed by parameter during class instantiation does not resolve to module class."""
        code = """
import src.calculator as calc

def test_fixture(calc):
    obj = calc.Calculator()
    obj.solve()
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow_alias_class.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.Calculator", dep.referenced_symbols)
        self.assertNotIn("src.calculator.Calculator.solve", dep.referenced_symbols)
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("parameter 'calc.Calculator'" in r for r in dep.uncertainty_reasons)
            or any("Unresolved instance method call on 'obj.solve'" in r for r in dep.uncertainty_reasons)
        )

    def test_package_root_constructor_shadowed_by_parameter(self) -> None:
        """Verify package root shadowed by parameter during class instantiation does not resolve to package class."""
        code = """
import src.calculator

def test_fixture(src):
    obj = src.calculator.Calculator()
    obj.solve()
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow_root_class.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.Calculator", dep.referenced_symbols)
        self.assertNotIn("src.calculator.Calculator.solve", dep.referenced_symbols)
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(
            any("parameter 'src.calculator'" in r for r in dep.uncertainty_reasons)
            or any("Unresolved instance method call on 'obj.solve'" in r for r in dep.uncertainty_reasons)
        )

    def test_local_variable_shadows_imported_function(self) -> None:
        """Finding 2: Local variable assignment must shadow imported function binding."""
        code = """
from src.calculator import add

def test_add():
    add = 42
    assert add == 42
"""
        extractor = TestSymbolExtractor(code, "tests/test_shadow_local.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)

    def test_local_variable_shadowing_various_assignment_forms(self) -> None:
        """Finding 2: Test augmented assignments, loop targets, and with-statement targets shadow imports."""
        code = """
from src.calculator import add, subtract, multiply, divide

def test_assignments():
    add = 1
    add += 2

    for subtract in [1, 2, 3]:
        pass

    with open("dummy") as multiply:
        pass

    if (divide := 10):
        pass

    assert add == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_assignments_shadow.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)
        self.assertNotIn("src.calculator.subtract", dep.referenced_symbols)
        self.assertNotIn("src.calculator.multiply", dep.referenced_symbols)
        self.assertNotIn("src.calculator.divide", dep.referenced_symbols)

    def test_nested_function_locals_do_not_leak_to_enclosing_scope(self) -> None:
        """Finding 2: Nested function local variables must not shadow outer test function scope."""
        code = """
from src.calculator import add

def test_nested_local():
    def helper():
        add = 99
        return add
    res = add(1, 2)
    assert res == 3
"""
        extractor = TestSymbolExtractor(code, "tests/test_nested_local.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # In test_nested_local, add is called directly and must resolve to src.calculator.add
        self.assertIn("src.calculator.add", dep.referenced_symbols)

    def test_ordinary_expressions_do_not_trigger_uncertainty(self) -> None:
        """Finding 3: f-strings, arithmetic, string concatenation, and len() must not trigger uncertainty."""
        code = """
import src.calculator

def test_safe_expressions():
    msg = f"Result: {10}"
    cleaned = msg.strip()

    combined = "foo" + "bar"
    upper = combined.upper()

    num = 10 + 5
    bits = num.bit_length()

    neg = -num
    neg_bits = neg.bit_length()

    length = len([1, 2, 3])
    length_bits = length.bit_length()

    assert cleaned and upper and bits and neg_bits and length_bits
"""
        extractor = TestSymbolExtractor(code, "tests/test_safe_expressions.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertFalse(dep.has_uncertainty)
        self.assertEqual(dep.uncertainty_reasons, [])

    def test_genuine_uncertainty_preserved_for_unknown_factory_call(self) -> None:
        """Finding 3: Genuine uncertainty must still be flagged for factory calls or unknown objects."""
        code = """
import src.calculator

def test_factory():
    service = create_unknown_service()
    service.do_something()
"""
        extractor = TestSymbolExtractor(code, "tests/test_factory_uncertainty.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertTrue(dep.has_uncertainty)
        self.assertTrue(any("create_unknown_service" in r or "service.do_something" in r for r in dep.uncertainty_reasons))

    def test_comprehension_target_does_not_leak_to_enclosing_scope(self) -> None:
        """Comprehension iteration targets (list/dict/set/gen) must not shadow imported functions in outer scope."""
        code = """
from src.calculator import add

def test_comprehension():
    results = [add for add in [1, 2, 3]]
    res = add(10, 20)
    assert res == 30
"""
        extractor = TestSymbolExtractor(code, "tests/test_comp_leak.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # In Python 3, comprehension target 'add' does not leak into enclosing function scope;
        # outer call add(10, 20) must resolve to src.calculator.add!
        self.assertIn("src.calculator.add", dep.referenced_symbols)

    def test_assignment_after_reference_shadows_throughout_function(self) -> None:
        """In Python lexical scoping, assigning to a name anywhere in a function makes it local throughout."""
        code = """
from src.calculator import add

def test_assign_after():
    res = add
    add = 42
    assert add == 42
"""
        extractor = TestSymbolExtractor(code, "tests/test_assign_after.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # Even though 'res = add' comes before 'add = 42', 'add' is a local variable for the entire scope
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)

    def test_global_declaration_excludes_name_from_locals(self) -> None:
        """Names explicitly declared 'global' must not be recorded as local variables."""
        import ast
        from services.test_symbol_extractor import _extract_assigned_locals

        code = """
from src.calculator import add

def test_global():
    global add
    add = 42
    res = add
    assert res == 42
"""
        extractor = TestSymbolExtractor(code, "tests/test_global_scope.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # In test_global, 'add' is declared global, so loading it resolves to imported symbol 'src.calculator.add'
        self.assertIn("src.calculator.add", dep.referenced_symbols)
        locals_ = _extract_assigned_locals(ast.parse(code).body[1].body)
        self.assertNotIn("add", locals_)

    def test_nonlocal_declaration_excludes_name_from_locals(self) -> None:
        """Names explicitly declared 'nonlocal' in nested functions must not be recorded as local to that scope."""
        import ast
        from services.test_symbol_extractor import _extract_assigned_locals

        code = """
def test_outer():
    count = 0
    def inner():
        nonlocal count
        count = 1
"""
        tree = ast.parse(code)
        inner_func = tree.body[0].body[1]
        locals_ = _extract_assigned_locals(inner_func.body)
        self.assertNotIn("count", locals_)

    def test_nested_function_import_does_not_leak_to_outer_test(self) -> None:
        """A nested function importing a symbol must NOT leak that symbol binding to the outer test."""
        code = """
def test_leak():
    def helper():
        from src.calculator import add
        val = add(1, 2)
    # Outside helper, add must not resolve to src.calculator.add
    val = add
"""
        extractor = TestSymbolExtractor(code, "tests/test_nested_import_leak.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # Inside helper, add(1, 2) was called, but when helper exited, symbol_bindings was restored.
        # Outer 'val = add' must not resolve as an outer dependency.
        # Specifically test with a symbol only imported inside helper but not called in helper:
        code_pure_leak = """
def test_pure_leak():
    def helper():
        from src.calculator import multiply
    val = multiply
"""
        dep_pure = TestSymbolExtractor(code_pure_leak, "tests/test_pure_leak.py").extract()[0]
        self.assertNotIn("src.calculator.multiply", dep_pure.referenced_symbols)

    def test_nested_function_assignment_does_not_overwrite_outer_instance(self) -> None:
        """A nested function assigning an instance variable must NOT overwrite the outer test's instance binding."""
        code = """
from src.calculator import Calculator

def test_overwrite():
    calc = Calculator()
    def helper():
        calc = "string literal"
        calc.strip()
    calc.solve()
"""
        extractor = TestSymbolExtractor(code, "tests/test_nested_overwrite.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # calc.solve() in the outer test must successfully resolve to Calculator.solve
        self.assertIn("src.calculator.Calculator.solve", dep.referenced_symbols)

    def test_outer_imported_bindings_remain_available_in_nested_function(self) -> None:
        """An outer test's imported bindings must remain available in nested functions where lexical scoping permits."""
        code = """
from src.calculator import Calculator

def test_outer_available():
    calc = Calculator()
    def helper():
        calc.solve()
    helper()
"""
        extractor = TestSymbolExtractor(code, "tests/test_outer_available.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.Calculator.solve", dep.referenced_symbols)

    def test_nested_function_import_forms_shadow_outer_bindings(self) -> None:
        """Finding 1: Local imports (import m, import m as a, from m import s, from m import s as a) shadow outer."""
        code = """
from src.calculator import add, subtract, multiply, divide

def test_outer():
    def helper():
        # Referencing before local import statements
        v1 = add
        v2 = subtract
        v3 = multiply
        v4 = divide

        import other_mod as add
        import other_pkg.sub
        from other_lib import symbol as subtract
        from other_source import multiply
        from other_module import divide
    helper()
"""
        extractor = TestSymbolExtractor(code, "tests/test_import_forms_shadow.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # None of the outer calculator symbols must be referenced because they are shadowed by helper's local imports
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)
        self.assertNotIn("src.calculator.subtract", dep.referenced_symbols)
        self.assertNotIn("src.calculator.multiply", dep.referenced_symbols)
        self.assertNotIn("src.calculator.divide", dep.referenced_symbols)

    def test_nested_reference_before_local_import_does_not_resolve_to_outer_symbol(self) -> None:
        """Finding 1: A nested function referencing a name before locally importing it must not resolve to outer symbol."""
        code = """
from src.calculator import add

def test_outer():
    def helper():
        x = add
        from src.other import add
    helper()
"""
        extractor = TestSymbolExtractor(code, "tests/test_ref_before_import.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertNotIn("src.calculator.add", dep.referenced_symbols)

    def test_function_default_argument_uses_enclosing_scope(self) -> None:
        """Finding 2: Function default argument expressions must resolve using the enclosing scope."""
        code = """
from src.calculator import add

def test_outer():
    def helper(add=add):
        return add
"""
        extractor = TestSymbolExtractor(code, "tests/test_default_arg.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)

    def test_lambda_default_argument_uses_enclosing_scope(self) -> None:
        """Finding 2: Lambda default argument expressions must resolve using the enclosing scope."""
        code = """
from src.calculator import add

def test_outer():
    f = lambda add=add: add
"""
        extractor = TestSymbolExtractor(code, "tests/test_lambda_default.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.add", dep.referenced_symbols)

    def test_nested_function_decorator_uses_enclosing_scope(self) -> None:
        """Finding 2: Function decorator expressions must resolve using the enclosing scope."""
        code = """
from src.calculator import Calculator

def test_outer():
    @register(Calculator)
    def helper(Calculator=1):
        return Calculator
"""
        extractor = TestSymbolExtractor(code, "tests/test_decorator_scope.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.Calculator", dep.referenced_symbols)

    def test_nested_function_parameter_annotation_uses_enclosing_scope(self) -> None:
        """Imported class used in a nested function's parameter annotation resolves in enclosing scope."""
        code = """
from src.calculator import Calculator

def test_outer():
    def helper(calc: Calculator):
        pass
"""
        extractor = TestSymbolExtractor(code, "tests/test_param_annotation.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.Calculator", dep.referenced_symbols)

    def test_nested_function_vararg_and_kwarg_annotations_use_enclosing_scope(self) -> None:
        """Imported classes used in *args and **kwargs annotations resolve in enclosing scope."""
        code = """
from src.calculator import Calculator, Operation

def test_outer():
    def helper(*args: Calculator, **kwargs: Operation):
        pass
"""
        extractor = TestSymbolExtractor(code, "tests/test_vararg_annotations.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        self.assertIn("src.calculator.Calculator", dep.referenced_symbols)
        self.assertIn("src.calculator.Operation", dep.referenced_symbols)

    def test_parameter_shadowing_symbol_does_not_shadow_its_own_annotation(self) -> None:
        """A parameter name that shadows an imported symbol inside the function body does not shadow the annotation."""
        code = """
from src.calculator import Calculator

def test_outer():
    def helper(Calculator: Calculator):
        return Calculator
"""
        extractor = TestSymbolExtractor(code, "tests/test_param_shadow_annotation.py")
        deps = extractor.extract()

        self.assertEqual(len(deps), 1)
        dep = deps[0]
        # Annotation ': Calculator' evaluated in enclosing scope resolves to src.calculator.Calculator
        self.assertIn("src.calculator.Calculator", dep.referenced_symbols)


if __name__ == "__main__":
    unittest.main()
