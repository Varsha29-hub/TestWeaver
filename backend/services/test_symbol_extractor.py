"""Test symbol dependency extractor for TestWeaver.

Week 3, Phase 3 — Stage 2:
Extracts fine-grained symbol references from test files using Python AST analysis.

Key Architectural Guarantees & Contracts:

1. Python Import Binding Semantics:
   - Root package bindings with multiple subpaths (e.g. 'import src.calculator' and 'import src.statistics').
   - Module aliases (e.g. 'import src.calculator as calc').
   - Symbol imports and aliases (e.g. 'from src.calc import add as plus').

2. Scope Isolation & Parameter Shadowing:
   - File-level imports form the baseline scope.
   - Function-local imports are strictly isolated to that test function and never
     leak into subsequent tests or mutate shared extractor structures.
   - Lexical Scopes & Parameters: Local parameter names shadow imported symbols, module
     aliases, and package root bindings. A parameter 'add' or 'calc' will never resolve to
     'src.calculator.add' or 'src.calculator.execute'.
   - Nested functions and lambdas maintain isolated parameter scopes using a scope stack
     so inner parameters never shadow names in the outer enclosing test function.

3. Syntactic Resolution vs. Verified Symbol Existence (Integration Contract):
   - Attribute chains traversed on imported modules or instances (e.g. 'calc.unknown_method()')
     are resolved syntactically into candidate symbol FQNs (e.g. 'src.calculator.unknown_method').
   - The extractor does NOT assume that an accessed attribute actually exists in the source module.
   - All such references are preserved in 'unverified_symbol_references' and registered in
     'module_attribute_references[module_path]' so the integration layer can cross-reference
     them against actual ChangedSymbol definitions or source ASTs. If an unverified attribute
     cannot be matched to a known symbol in the target module, the integration layer MUST
     apply conservative ImpactType.MODULE fallback rather than classifying the test as UNAFFECTED.

4. Relative Import Integration Contract:
   - Relative imports (e.g. 'from ..calculator import add') cannot be safely dot-stripped or
     resolved without package/repository anchor context.
   - The extractor never fabricates module paths. Instead, it explicitly marks
     'has_unresolved_relative_imports = True' and preserves relative module strings in
     'unresolved_relative_imports'.
   - The integration layer MUST conservatively classify a test as ImpactType.MODULE whenever
     unresolved relative imports could refer to changed Python modules and their targets cannot
     be ruled out.

5. Dynamic Import Detection (Including Aliases):
   - Detects dynamic imports through direct and aliased forms:
     * 'importlib.import_module(...)'
     * 'import importlib as il; il.import_module(...)'
     * 'from importlib import import_module; import_module(...)'
     * 'from importlib import import_module as imp; imp(...)'
     * '__import__(...)'
   - Flags uncertainty immediately to trigger conservative fallback.
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Builtin reflection functions that obscure static dependency resolution
DYNAMIC_REFLECTION_FUNCTIONS = {
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "eval",
    "exec",
    "__import__",
}

# Known builtin constructors whose instances should not trigger uncertainty
BUILTIN_CONSTRUCTORS = {
    "list": "__list__",
    "dict": "__dict__",
    "set": "__set__",
    "tuple": "__tuple__",
    "str": "__str__",
    "int": "__num__",
    "float": "__num__",
    "bool": "__bool__",
    "bytes": "__bytes__",
    "len": "__num__",
}

KNOWN_BUILTIN_LITERAL_TYPES = {
    "__str__",
    "__num__",
    "__bool__",
    "__list__",
    "__dict__",
    "__set__",
    "__tuple__",
    "__bytes__",
}


@dataclass
class Scope:
    """Lexical scope tracking parameters, local variables, and environment snapshots."""

    parameters: set[str] = field(default_factory=set)
    local_variables: set[str] = field(default_factory=set)
    local_imported_names: set[str] = field(default_factory=set)
    active_local_imports: set[str] = field(default_factory=set)
    saved_local_instances: dict[str, str] = field(default_factory=dict)
    saved_self_instances: dict[str, str] = field(default_factory=dict)
    saved_unknown_instances: set[str] = field(default_factory=set)
    saved_module_aliases: dict[str, str] = field(default_factory=dict)
    saved_symbol_bindings: dict[str, str] = field(default_factory=dict)
    saved_package_root_bindings: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class TestSymbolDependencies:
    """Represents symbol-level dependencies and uncertainty for an individual test."""

    node_id: str
    test_file: str
    test_name: str

    # Syntactically resolved symbol candidates (e.g. 'src.calculator.add')
    referenced_symbols: set[str] = field(default_factory=set)

    # Modules directly or indirectly imported by this test or its enclosing file
    imported_modules: set[str] = field(default_factory=set)

    # Uncertainty flags and detailed reasons triggering conservative fallback
    has_uncertainty: bool = False
    uncertainty_reasons: list[str] = field(default_factory=list)

    # Relative imports that could not be resolved without package anchor context
    has_unresolved_relative_imports: bool = False
    unresolved_relative_imports: list[str] = field(default_factory=list)

    # Syntactic attribute references on modules/instances requiring source verification
    has_unverified_symbol_references: bool = False
    unverified_symbol_references: set[str] = field(default_factory=set)
    module_attribute_references: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


def _normalize_path(path: str) -> str:
    """Normalize file path to POSIX format with stripped whitespace."""
    return path.strip().replace("\\", "/")


def _flatten_attribute_chain(
    node: ast.AST,
) -> tuple[list[str], Optional[ast.AST]]:
    """Flatten an AST Attribute chain into a list of identifier parts and the root node.

    For example:
        'src.calculator.add' -> (['src', 'calculator', 'add'], Name(id='src'))
        'calc.solve' -> (['calc', 'solve'], Name(id='calc'))
        'self.calc.solve' -> (['self', 'calc', 'solve'], Name(id='self'))
        'get_obj().solve' -> (['solve'], Call(...))
    """
    parts: list[str] = []
    curr = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value

    parts.reverse()
    if isinstance(curr, ast.Name):
        return [curr.id] + parts, curr
    return parts, curr


def _extract_target_names(target: ast.AST) -> set[str]:
    """Recursively extract bare Name identifiers from an assignment target."""
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_extract_target_names(elt))
    elif isinstance(target, ast.Starred):
        names.update(_extract_target_names(target.value))
    return names


def _extract_assigned_locals(stmts: list[ast.AST]) -> set[str]:
    """Extract all locally assigned variable names in a function body according to Python scoping.

    Excludes names explicitly declared 'global' or 'nonlocal'.
    Does NOT enter nested FunctionDef, AsyncFunctionDef, or ClassDef blocks.
    Comprehension loop targets do not leak into the enclosing function scope.
    """
    locals_found: set[str] = set()
    explicit_non_locals: set[str] = set()

    def _scan(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested function or class: strict boundary, do not scan
            return

        if isinstance(node, (ast.Global, ast.Nonlocal)):
            explicit_non_locals.update(node.names)
            return

        if isinstance(node, ast.Assign):
            for t in node.targets:
                locals_found.update(_extract_target_names(t))
        elif isinstance(node, ast.AnnAssign):
            locals_found.update(_extract_target_names(node.target))
        elif isinstance(node, ast.AugAssign):
            locals_found.update(_extract_target_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            locals_found.update(_extract_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    locals_found.update(_extract_target_names(item.optional_vars))
        elif isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.name:
                    locals_found.add(handler.name)
        elif isinstance(node, ast.NamedExpr):
            locals_found.update(_extract_target_names(node.target))

        # Recurse children, but do not enter nested functions, classes, or comprehension scopes for targets
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Inside comprehension, only walrus operators (NamedExpr) leak to enclosing scope.
                # Comprehension iteration targets (comp.generators) do NOT leak.
                for comp_child in ast.walk(child):
                    if isinstance(comp_child, ast.NamedExpr):
                        locals_found.update(_extract_target_names(comp_child.target))
                continue
            _scan(child)

    for stmt in stmts:
        _scan(stmt)

    return locals_found - explicit_non_locals


def _extract_imported_names(stmts: list[ast.AST]) -> set[str]:
    """Extract all names bound by import statements in a function body according to Python scoping.

    Does NOT enter nested FunctionDef, AsyncFunctionDef, or ClassDef blocks.
    """
    imported_names: set[str] = set()

    def _scan(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imported_names.add(alias.asname)
                else:
                    imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bound_name = alias.asname if alias.asname else alias.name
                    imported_names.add(bound_name)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            _scan(child)

    for stmt in stmts:
        _scan(stmt)

    return imported_names


def _get_literal_type(
    node: ast.AST,
    local_instances: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Identify if an AST node is a Python literal or safe expression, returning a synthetic type."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "__str__"
        if isinstance(node.value, bool):
            return "__bool__"
        if isinstance(node.value, (int, float)):
            return "__num__"
        if isinstance(node.value, bytes):
            return "__bytes__"
    elif isinstance(node, ast.JoinedStr):
        # f-strings always produce strings at runtime
        return "__str__"
    elif isinstance(node, ast.List):
        return "__list__"
    elif isinstance(node, ast.Dict):
        return "__dict__"
    elif isinstance(node, ast.Set):
        return "__set__"
    elif isinstance(node, ast.Tuple):
        return "__tuple__"
    elif isinstance(node, ast.Compare):
        # Any comparison evaluates to boolean
        return "__bool__"
    elif isinstance(node, ast.UnaryOp):
        operand_type = _get_literal_type(node.operand, local_instances)
        if isinstance(node.op, ast.Not):
            return "__bool__"
        if operand_type == "__num__" and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            return "__num__"
    elif isinstance(node, ast.BinOp):
        left_type = _get_literal_type(node.left, local_instances)
        right_type = _get_literal_type(node.right, local_instances)
        if isinstance(node.op, ast.Add):
            if left_type == "__str__" and right_type == "__str__":
                return "__str__"
            if left_type == "__num__" and right_type == "__num__":
                return "__num__"
            if left_type == "__list__" and right_type == "__list__":
                return "__list__"
            if left_type == "__tuple__" and right_type == "__tuple__":
                return "__tuple__"
            if left_type == "__bytes__" and right_type == "__bytes__":
                return "__bytes__"
        elif isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Pow)):
            if left_type == "__num__" and right_type == "__num__":
                return "__num__"
        elif isinstance(node.op, ast.Mult):
            if left_type == "__num__" and right_type == "__num__":
                return "__num__"
            if (left_type == "__str__" and right_type == "__num__") or (left_type == "__num__" and right_type == "__str__"):
                return "__str__"
            if (left_type == "__list__" and right_type == "__num__") or (left_type == "__num__" and right_type == "__list__"):
                return "__list__"
        elif isinstance(node.op, ast.Mod):
            if left_type == "__str__":
                return "__str__"
            if left_type == "__num__" and right_type == "__num__":
                return "__num__"
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "str":
                return "__str__"
            elif func_name in ("len", "int", "float"):
                return "__num__"
            elif func_name == "bool":
                return "__bool__"
            elif func_name == "list":
                return "__list__"
            elif func_name == "dict":
                return "__dict__"
            elif func_name == "set":
                return "__set__"
            elif func_name == "tuple":
                return "__tuple__"
            elif func_name == "bytes":
                return "__bytes__"
    elif isinstance(node, ast.Name):
        if local_instances and node.id in local_instances:
            inst = local_instances[node.id]
            if inst in KNOWN_BUILTIN_LITERAL_TYPES:
                return inst
    return None


class TestSymbolExtractor:
    """Parses Python test source code to extract symbol references per test."""

    def __init__(self, file_content: str, file_path: str) -> None:
        self.file_content = file_content
        self.file_path = _normalize_path(file_path)

        # File-level import structures (immutable baseline for test function scopes)
        self.package_root_bindings: dict[str, set[str]] = defaultdict(set)
        self.module_aliases: dict[str, str] = {}
        self.symbol_bindings: dict[str, str] = {}
        self.file_imported_modules: set[str] = set()

        # File-level uncertainty flags
        self.file_has_uncertainty: bool = False
        self.file_uncertainty_reasons: list[str] = []
        self.file_has_unresolved_relative_imports: bool = False
        self.file_unresolved_relative_imports: list[str] = []

    def extract(self) -> list[TestSymbolDependencies]:
        """Extract symbol dependencies for all discovered tests in the file."""
        if not self.file_content.strip():
            return []

        try:
            tree = ast.parse(self.file_content, filename=self.file_path)
        except SyntaxError as e:
            logger.warning(
                "Syntax error parsing test file %s: %s", self.file_path, e
            )
            return []

        # Pass 1: Collect file-level imports and module-level statements
        self._collect_file_imports(tree)

        # Pass 2: Extract tests from top-level functions and test classes
        results: list[TestSymbolDependencies] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_test_function_name(node.name):
                    dep = self._analyze_test_function(
                        func_node=node,
                        class_name=None,
                    )
                    results.append(dep)
            elif isinstance(node, ast.ClassDef):
                if self._is_test_class_name(node.name):
                    class_deps = self._analyze_test_class(node)
                    results.extend(class_deps)

        return results

    def _is_test_function_name(self, name: str) -> bool:
        return name.startswith("test_") or name.endswith("_test")

    def _is_test_class_name(self, name: str) -> bool:
        return name.startswith("Test") or name.endswith("Test")

    def _collect_file_imports(self, tree: ast.Module) -> None:
        """Scan top-level statements for imports, aliases, and wildcards."""
        for node in tree.body:
            if isinstance(node, ast.Import):
                self._record_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._record_import_from(node)

    def _record_import(self, node: ast.Import) -> None:
        """Process 'import pkg.sub [as alias]' statements."""
        for alias in node.names:
            imported_name = alias.name
            self.file_imported_modules.add(imported_name)

            if alias.asname:
                self.module_aliases[alias.asname] = imported_name
            else:
                # Python binds the root package identifier
                root_pkg = imported_name.split(".")[0]
                self.package_root_bindings[root_pkg].add(imported_name)

    def _record_import_from(self, node: ast.ImportFrom) -> None:
        """Process 'from pkg.sub import name [as alias]' and relative/wildcard imports."""
        module_path = node.module or ""

        if node.level > 0:
            # Relative import: Cannot safely resolve without package context
            self.file_has_uncertainty = True
            self.file_has_unresolved_relative_imports = True
            if module_path:
                rel_imports = [f"{'.' * node.level}{module_path}"]
            else:
                rel_imports = [
                    f"{'.' * node.level}{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                ]
                if not rel_imports:
                    rel_imports = ["." * node.level]
            for rel_import in rel_imports:
                self.file_unresolved_relative_imports.append(rel_import)
                self.file_uncertainty_reasons.append(
                    f"Relative import 'from {rel_import}' requires package context to resolve"
                )
            return

        if module_path:
            self.file_imported_modules.add(module_path)

        for alias in node.names:
            if alias.name == "*":
                # Wildcard import obscures symbol source
                self.file_has_uncertainty = True
                self.file_uncertainty_reasons.append(
                    f"Wildcard import 'from {module_path} import *' detected"
                )
            else:
                target_fqn = f"{module_path}.{alias.name}" if module_path else alias.name
                bound_name = alias.asname if alias.asname else alias.name
                self.symbol_bindings[bound_name] = target_fqn
                # An imported name could also be a submodule (e.g. 'from src import calc')
                self.module_aliases[bound_name] = target_fqn
                self.file_imported_modules.add(target_fqn)

    def _analyze_test_class(
        self, class_node: ast.ClassDef
    ) -> list[TestSymbolDependencies]:
        """Analyze test methods inside a test class, capturing shared setUp state."""
        class_name = class_node.name
        self_instances: dict[str, str] = {}

        # Look for setUp / async setUp methods to harvest self.<attr> class instantiations
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in ("setUp", "asyncSetUp", "setup_method"):
                    self._harvest_self_instances(item, self_instances)

        results: list[TestSymbolDependencies] = []
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_test_function_name(item.name):
                    dep = self._analyze_test_function(
                        func_node=item,
                        class_name=class_name,
                        initial_self_instances=dict(self_instances),
                    )
                    results.append(dep)

        return results

    def _harvest_self_instances(
        self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], out_map: dict[str, str]
    ) -> None:
        """Scan a setup method for 'self.<attr> = Class()' assignments."""
        for stmt in ast.walk(func_node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == "self":
                            class_fqn = self._resolve_call_to_class(
                                stmt.value,
                                self.symbol_bindings,
                                self.module_aliases,
                                self.package_root_bindings,
                            )
                            if class_fqn:
                                out_map[target.attr] = class_fqn
                            else:
                                out_map[target.attr] = "__unknown__"

    def _resolve_call_to_class(
        self,
        node: ast.AST,
        symbol_bindings: dict[str, str],
        module_aliases: dict[str, str],
        package_root_bindings: dict[str, set[str]],
    ) -> Optional[str]:
        """Determine if an AST node is a Call instantiating a known class."""
        if not isinstance(node, ast.Call):
            return None

        # Resolve func expression
        parts, root_node = _flatten_attribute_chain(node.func)
        if not parts:
            return None

        # Check for builtin constructor
        if len(parts) == 1 and parts[0] in BUILTIN_CONSTRUCTORS:
            return BUILTIN_CONSTRUCTORS[parts[0]]

        # Case 1: Simple Name (e.g. 'Calculator()')
        if len(parts) == 1:
            name = parts[0]
            if name in symbol_bindings:
                return symbol_bindings[name]

        # Case 2: Aliased module access (e.g. 'calc.Calculator()')
        first = parts[0]
        if first in module_aliases:
            mod_prefix = module_aliases[first]
            remainder = ".".join(parts[1:])
            return f"{mod_prefix}.{remainder}" if remainder else mod_prefix

        # Case 3: Package root binding (e.g. 'src.calc.Calculator()')
        if first in package_root_bindings:
            subpaths = sorted(
                package_root_bindings[first], key=len, reverse=True
            )
            for subpath in subpaths:
                sub_parts = subpath.split(".")
                if parts[: len(sub_parts)] == sub_parts:
                    remainder = parts[len(sub_parts) :]
                    if remainder:
                        return f"{subpath}.{'.'.join(remainder)}"
                    return subpath

        return None

    def _analyze_test_function(
        self,
        func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        class_name: Optional[str] = None,
        initial_self_instances: Optional[dict[str, str]] = None,
    ) -> TestSymbolDependencies:
        """Analyze an individual test function or method in complete isolation."""
        func_name = func_node.name
        if class_name:
            node_id = f"{self.file_path}::{class_name}::{func_name}"
        else:
            node_id = f"{self.file_path}::{func_name}"

        dep = TestSymbolDependencies(
            node_id=node_id,
            test_file=self.file_path,
            test_name=func_name,
            imported_modules=set(self.file_imported_modules),
            has_uncertainty=self.file_has_uncertainty,
            uncertainty_reasons=list(self.file_uncertainty_reasons),
            has_unresolved_relative_imports=self.file_has_unresolved_relative_imports,
            unresolved_relative_imports=list(self.file_unresolved_relative_imports),
        )

        visitor = _TestFunctionVisitor(
            extractor=self,
            dependencies=dep,
            initial_self_instances=initial_self_instances or {},
        )
        visitor.visit(func_node)

        return dep


class _TestFunctionVisitor(ast.NodeVisitor):
    """AST visitor traversing a single test function body to record symbol references."""

    def __init__(
        self,
        extractor: TestSymbolExtractor,
        dependencies: TestSymbolDependencies,
        initial_self_instances: dict[str, str],
    ) -> None:
        self.extractor = extractor
        self.dep = dependencies

        # Isolated local copies of import mappings for this test scope.
        # Function-local imports mutate ONLY these dictionaries and never the shared extractor.
        self.package_root_bindings: dict[str, set[str]] = defaultdict(
            set,
            {k: set(v) for k, v in extractor.package_root_bindings.items()},
        )
        self.module_aliases: dict[str, str] = dict(extractor.module_aliases)
        self.symbol_bindings: dict[str, str] = dict(extractor.symbol_bindings)

        # Local instance bindings (variable_name -> class_fqn or literal_type or "__unknown__")
        self.local_instances: dict[str, str] = {}
        self.self_instances: dict[str, str] = dict(initial_self_instances)
        self.unknown_instances: set[str] = set()

        # Scope stack of Scope objects for lexical scope tracking
        self.scope_stack: list[Scope] = []

    def _extract_params(self, args: ast.arguments) -> set[str]:
        """Extract parameter names from an ast.arguments object."""
        params = {
            arg.arg
            for arg in args.args + args.posonlyargs + args.kwonlyargs
        }
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)
        return params

    def _is_local_or_parameter(self, name: str) -> bool:
        """Check if name resolves to a parameter, locally assigned variable, or unexecuted import in lexical scope."""
        for scope in reversed(self.scope_stack):
            if name in scope.parameters or name in scope.local_variables:
                return True
            if name in scope.local_imported_names:
                if name not in scope.active_local_imports:
                    return True
                return False
        return False

    def _is_local_variable(self, name: str) -> bool:
        """Check if name is a locally assigned variable or unexecuted import in lexical scope."""
        for scope in reversed(self.scope_stack):
            if name in scope.local_variables:
                return True
            if name in scope.local_imported_names:
                if name not in scope.active_local_imports:
                    return True
                return False
        return False

    def _is_parameter(self, name: str) -> bool:
        """Check if name is a parameter in any currently active enclosing scope."""
        for scope in reversed(self.scope_stack):
            if name in scope.parameters:
                return True
        return False

    @property
    def parameter_names(self) -> set[str]:
        """Return all parameters across currently active scopes."""
        res: set[str] = set()
        for s in self.scope_stack:
            res.update(s.parameters)
        return res

    def _resolve_call_to_class(self, node: ast.AST) -> Optional[str]:
        if not isinstance(node, ast.Call):
            return None

        parts, _ = _flatten_attribute_chain(node.func)
        if parts:
            first = parts[0]
            # Parameter and local names in active lexical scopes shadow imported names and constructors
            if first != "self" and self._is_local_or_parameter(first):
                return None

        return self.extractor._resolve_call_to_class(
            node,
            self.symbol_bindings,
            self.module_aliases,
            self.package_root_bindings,
        )

    def _enter_scope(self, args: ast.arguments, body: list[ast.AST]) -> None:
        params = self._extract_params(args)
        locals_ = _extract_assigned_locals(body)
        imported_ = _extract_imported_names(body)
        scope = Scope(
            parameters=params,
            local_variables=locals_,
            local_imported_names=imported_,
            active_local_imports=set(),
            saved_local_instances=dict(self.local_instances),
            saved_self_instances=dict(self.self_instances),
            saved_unknown_instances=set(self.unknown_instances),
            saved_module_aliases=dict(self.module_aliases),
            saved_symbol_bindings=dict(self.symbol_bindings),
            saved_package_root_bindings={
                k: set(v) for k, v in self.package_root_bindings.items()
            },
        )
        self.scope_stack.append(scope)

    def _exit_scope(self) -> None:
        scope = self.scope_stack.pop()
        self.local_instances = scope.saved_local_instances
        self.self_instances = scope.saved_self_instances
        self.unknown_instances = scope.saved_unknown_instances
        self.module_aliases = scope.saved_module_aliases
        self.symbol_bindings = scope.saved_symbol_bindings
        self.package_root_bindings = defaultdict(set, scope.saved_package_root_bindings)

    def _visit_function_definition(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        # 1. Definition-time expressions are evaluated in the ENCLOSING scope:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        for arg in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        ):
            if arg.annotation:
                self.visit(arg.annotation)
        if node.args.vararg and node.args.vararg.annotation:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        # 2. Enter function scope for function body:
        self._enter_scope(node.args, node.body)
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self._exit_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # 1. Default expressions are evaluated in the ENCLOSING scope:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)

        # 2. Enter lambda scope for lambda body:
        self._enter_scope(node.args, [ast.Expr(value=node.body)])
        try:
            self.visit(node.body)
        finally:
            self._exit_scope()

    def visit_Import(self, node: ast.Import) -> None:
        # Local import isolated inside this test function
        for alias in node.names:
            self.dep.imported_modules.add(alias.name)
            if alias.asname:
                self.module_aliases[alias.asname] = alias.name
                bound_name = alias.asname
            else:
                root_pkg = alias.name.split(".")[0]
                self.package_root_bindings[root_pkg].add(alias.name)
                bound_name = root_pkg
            if self.scope_stack:
                self.scope_stack[-1].active_local_imports.add(bound_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Local from-import isolated inside this test function
        module_path = node.module or ""
        if node.level > 0:
            if self.scope_stack:
                for alias in node.names:
                    if alias.name != "*":
                        bound_name = alias.asname if alias.asname else alias.name
                        self.scope_stack[-1].active_local_imports.add(bound_name)
            self.dep.has_uncertainty = True
            self.dep.has_unresolved_relative_imports = True
            if module_path:
                rel_imports = [f"{'.' * node.level}{module_path}"]
            else:
                rel_imports = [
                    f"{'.' * node.level}{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                ]
                if not rel_imports:
                    rel_imports = ["." * node.level]
            for rel_import in rel_imports:
                self.dep.unresolved_relative_imports.append(rel_import)
                self.dep.uncertainty_reasons.append(
                    f"Local relative import 'from {rel_import}' requires package context"
                )
            return

        if module_path:
            self.dep.imported_modules.add(module_path)

        for alias in node.names:
            if alias.name == "*":
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Local wildcard import from '{module_path}'"
                )
            else:
                target_fqn = f"{module_path}.{alias.name}" if module_path else alias.name
                bound_name = alias.asname if alias.asname else alias.name
                self.symbol_bindings[bound_name] = target_fqn
                self.module_aliases[bound_name] = target_fqn
                self.dep.imported_modules.add(target_fqn)
                if self.scope_stack:
                    self.scope_stack[-1].active_local_imports.add(bound_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._track_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self._track_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._track_assignment([node.target], node.value)
        self.generic_visit(node)

    def _track_assignment(self, targets: list[ast.AST], value_node: ast.AST) -> None:
        """Track variable assignments to known classes, literals, or unknown instances."""
        class_fqn = self._resolve_call_to_class(value_node)
        literal_type = _get_literal_type(value_node, self.local_instances)

        for target in targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if class_fqn:
                    self.local_instances[var_name] = class_fqn
                elif literal_type:
                    self.local_instances[var_name] = literal_type
                elif isinstance(value_node, ast.Name) and value_node.id in self.local_instances:
                    self.local_instances[var_name] = self.local_instances[value_node.id]
                else:
                    # Unresolved factory call, helper return value, or unknown object
                    self.unknown_instances.add(var_name)
                    self.local_instances[var_name] = "__unknown__"
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    attr_name = target.attr
                    if class_fqn:
                        self.self_instances[attr_name] = class_fqn
                    elif literal_type:
                        self.self_instances[attr_name] = literal_type
                    else:
                        self.self_instances[attr_name] = "__unknown__"

    def visit_Call(self, node: ast.Call) -> None:
        # 1. Detect dynamic reflection calls (including aliased importlib)
        self._check_dynamic_reflection(node.func)

        # 2. Resolve target callee symbol
        self._resolve_reference(node.func)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Non-call attribute access (e.g. reading a class attribute or module constant)
        self._resolve_reference(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Direct name references (e.g. passing a function as a callback or calling it)
        if isinstance(node.ctx, ast.Load):
            # Parameter and locally assigned names shadow imported symbols in the current lexical scope
            if node.id != "self" and self._is_local_or_parameter(node.id):
                self.generic_visit(node)
                return

            if node.id in DYNAMIC_REFLECTION_FUNCTIONS:
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Dynamic reflection function referenced: '{node.id}'"
                )
            elif node.id in self.symbol_bindings:
                bound_target = self.symbol_bindings[node.id]
                if bound_target.startswith("importlib"):
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Dynamic import call: '{node.id}' bound to '{bound_target}'"
                    )
                else:
                    self.dep.referenced_symbols.add(bound_target)

        self.generic_visit(node)

    def _check_dynamic_reflection(self, func_node: ast.AST) -> None:
        """Flag dynamic reflection patterns like getattr, eval, or aliased importlib."""
        if isinstance(func_node, ast.Name):
            name = func_node.id
            if name != "self" and self._is_local_or_parameter(name):
                return
            if name in DYNAMIC_REFLECTION_FUNCTIONS:
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Dynamic reflection call: '{name}()'"
                )
            elif name in self.symbol_bindings and self.symbol_bindings[name].startswith("importlib"):
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Dynamic import call: '{name}()' bound to '{self.symbol_bindings[name]}'"
                )
        elif isinstance(func_node, ast.Attribute):
            parts, _ = _flatten_attribute_chain(func_node)
            if parts:
                first = parts[0]
                if first != "self" and self._is_local_or_parameter(first):
                    return
                # Direct 'importlib.import_module()'
                if first == "importlib":
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Dynamic import call: '{'.'.join(parts)}()'"
                    )
                # Aliased 'il.import_module()'
                elif first in self.module_aliases and self.module_aliases[first].startswith("importlib"):
                    resolved_chain = [self.module_aliases[first]] + parts[1:]
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Dynamic import call: '{'.'.join(parts)}()' (aliased from '{'.'.join(resolved_chain)}')"
                    )
                # Bound symbol attribute 'imp_lib.util.find_spec()'
                elif first in self.symbol_bindings and self.symbol_bindings[first].startswith("importlib"):
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Dynamic import call: '{'.'.join(parts)}()' (bound to '{self.symbol_bindings[first]}')"
                    )

    def _resolve_reference(self, node: ast.AST) -> None:
        """Resolve an AST node (Attribute or Name) to candidate symbol FQNs.

        Preserves unverified attribute access information for cross-referencing by the
        integration layer against source symbols.
        """
        parts, root_node = _flatten_attribute_chain(node)
        if not parts:
            return

        first = parts[0]

        # Case 1: Method call on 'self' (e.g. 'self.calc.solve()')
        if first == "self" and len(parts) >= 2:
            second = parts[1]
            if second in self.self_instances:
                class_fqn = self.self_instances[second]
                if class_fqn == "__unknown__":
                    if self.dep.imported_modules:
                        self.dep.has_uncertainty = True
                        self.dep.uncertainty_reasons.append(
                            f"Unresolved self instance method call on 'self.{second}.{parts[2] if len(parts) >= 3 else ''}'"
                        )
                    return
                elif class_fqn not in KNOWN_BUILTIN_LITERAL_TYPES:
                    remainder = parts[2:]
                    if remainder:
                        sym_fqn = f"{class_fqn}.{'.'.join(remainder)}"
                        self.dep.referenced_symbols.add(sym_fqn)
                        # Mark as unverified attribute access on class
                        self.dep.has_unverified_symbol_references = True
                        self.dep.unverified_symbol_references.add(sym_fqn)
                        mod_part = class_fqn.rsplit(".", 1)[0]
                        attr_part = f"{class_fqn.rsplit('.', 1)[-1]}.{'.'.join(remainder)}"
                        self.dep.module_attribute_references[mod_part].add(attr_part)
                    else:
                        self.dep.referenced_symbols.add(class_fqn)
                return
            elif len(parts) >= 3 and self.dep.imported_modules:
                # 'self.unresolved.method()' with imported modules -> flag uncertainty
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Unresolved self attribute call: 'self.{second}.{parts[2]}'"
                )
                return

        # Case 0: Local parameter names shadow imported symbols, module aliases, and package roots.
        if first != "self" and self._is_parameter(first):
            if first in self.local_instances:
                class_fqn = self.local_instances[first]
                if class_fqn == "__unknown__":
                    if self.dep.imported_modules:
                        self.dep.has_uncertainty = True
                        self.dep.uncertainty_reasons.append(
                            f"Unresolved instance method call on parameter '{first}.{parts[1] if len(parts) >= 2 else ''}'"
                        )
                    return
                elif class_fqn not in KNOWN_BUILTIN_LITERAL_TYPES:
                    remainder = parts[1:]
                    if remainder:
                        sym_fqn = f"{class_fqn}.{'.'.join(remainder)}"
                        self.dep.referenced_symbols.add(sym_fqn)
                        self.dep.has_unverified_symbol_references = True
                        self.dep.unverified_symbol_references.add(sym_fqn)
                        mod_part = class_fqn.rsplit(".", 1)[0]
                        attr_part = f"{class_fqn.rsplit('.', 1)[-1]}.{'.'.join(remainder)}"
                        self.dep.module_attribute_references[mod_part].add(attr_part)
                    else:
                        self.dep.referenced_symbols.add(class_fqn)
                    return
            else:
                # Parameter was not assigned to a known class instance
                if len(parts) >= 2 and self.dep.imported_modules:
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Unresolved instance method call on parameter '{first}.{parts[1]}'"
                    )
                return

        # Case 2: Method call on local instance variable (e.g. 'calculator.solve()')
        if first in self.local_instances:
            class_fqn = self.local_instances[first]
            if class_fqn == "__unknown__":
                if self.dep.imported_modules:
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Unresolved instance method call on '{first}.{parts[1] if len(parts) >= 2 else ''}'"
                    )
                return
            elif class_fqn not in KNOWN_BUILTIN_LITERAL_TYPES:
                remainder = parts[1:]
                if remainder:
                    sym_fqn = f"{class_fqn}.{'.'.join(remainder)}"
                    self.dep.referenced_symbols.add(sym_fqn)
                    # Mark as unverified attribute access on class
                    self.dep.has_unverified_symbol_references = True
                    self.dep.unverified_symbol_references.add(sym_fqn)
                    mod_part = class_fqn.rsplit(".", 1)[0]
                    attr_part = f"{class_fqn.rsplit('.', 1)[-1]}.{'.'.join(remainder)}"
                    self.dep.module_attribute_references[mod_part].add(attr_part)
                else:
                    self.dep.referenced_symbols.add(class_fqn)
            return

        # Case 2b: Locally assigned variable without instance call (shadows imports)
        if first != "self" and self._is_local_variable(first):
            return

        # Case 3: Aliased module access (e.g. 'calc.add()' or 'calc.Calculator.solve()')
        if first in self.module_aliases:
            mod_prefix = self.module_aliases[first]
            remainder = parts[1:]
            if remainder:
                sym_fqn = f"{mod_prefix}.{'.'.join(remainder)}"
                self.dep.referenced_symbols.add(sym_fqn)
                # Mark as unverified attribute access on module
                self.dep.has_unverified_symbol_references = True
                self.dep.unverified_symbol_references.add(sym_fqn)
                self.dep.module_attribute_references[mod_prefix].add(".".join(remainder))
            else:
                self.dep.referenced_symbols.add(mod_prefix)
            return

        # Case 4: Directly imported symbol (e.g. 'Calculator.solve()' or 'add()')
        if first in self.symbol_bindings:
            sym_prefix = self.symbol_bindings[first]
            remainder = parts[1:]
            if remainder:
                sym_fqn = f"{sym_prefix}.{'.'.join(remainder)}"
                self.dep.referenced_symbols.add(sym_fqn)
                # Mark as unverified attribute access on imported symbol
                self.dep.has_unverified_symbol_references = True
                self.dep.unverified_symbol_references.add(sym_fqn)
                mod_part = sym_prefix.rsplit(".", 1)[0]
                attr_part = f"{sym_prefix.rsplit('.', 1)[-1]}.{'.'.join(remainder)}"
                self.dep.module_attribute_references[mod_part].add(attr_part)
            else:
                self.dep.referenced_symbols.add(sym_prefix)
            return

        # Case 5: Package root dotted chain (e.g. 'src.calculator.add()' or 'src.statistics.mean()')
        if first in self.package_root_bindings:
            candidate_subpaths = sorted(
                self.package_root_bindings[first],
                key=len,
                reverse=True,
            )
            matched = False
            for subpath in candidate_subpaths:
                sub_parts = subpath.split(".")
                if parts[: len(sub_parts)] == sub_parts:
                    remainder = parts[len(sub_parts) :]
                    if remainder:
                        sym_fqn = f"{subpath}.{'.'.join(remainder)}"
                        self.dep.referenced_symbols.add(sym_fqn)
                        # Mark as unverified attribute access on module
                        self.dep.has_unverified_symbol_references = True
                        self.dep.unverified_symbol_references.add(sym_fqn)
                        self.dep.module_attribute_references[subpath].add(".".join(remainder))
                    else:
                        self.dep.referenced_symbols.add(subpath)
                    matched = True
                    break

            if not matched:
                # Starts with known root package, but subpath was not imported
                self.dep.has_uncertainty = True
                self.dep.uncertainty_reasons.append(
                    f"Unresolved subpath under package root '{first}': '{'.'.join(parts)}'"
                )
            return

        # Case 6: Attribute call on an unresolvable variable, parameter, or unknown instance
        if len(parts) >= 2:
            if (
                first in self.unknown_instances
                or self._is_parameter(first)
                or not isinstance(root_node, ast.Name)
            ):
                if self.dep.imported_modules:
                    self.dep.has_uncertainty = True
                    self.dep.uncertainty_reasons.append(
                        f"Unresolved instance method call on '{first}.{parts[1]}'"
                    )
