"""Unit tests for the pure helper functions shared by the finetuning pipelines.

``finetuning``, ``finetuning_v2`` and ``finetuning_v3`` each expose a nearly
identical set of dependency-free helpers (a safe math parser, JSON recovery,
AST security checks, literal evaluation, ...).  The tests below are
parametrized so that every version is exercised with the same assertions.
"""

import ast
import importlib
import math

import pytest

MODULE_NAMES = ["finetuning", "finetuning_v2", "finetuning_v3"]


@pytest.fixture(params=MODULE_NAMES)
def mod(request):
    return importlib.import_module(request.param)


# ---------------------------------------------------------------------------
# calculate_expression
# ---------------------------------------------------------------------------

def test_calculate_expression_basic_arithmetic(mod):
    assert mod.calculate_expression("2 + 3 * 4") == "14.0"
    assert mod.calculate_expression("(2 + 3) * 4") == "20.0"
    assert mod.calculate_expression("-5 + 2") == "-3.0"


def test_calculate_expression_supports_named_constants(mod):
    assert mod.calculate_expression("pi") == str(math.pi)
    assert mod.calculate_expression("e") == str(math.e)


def test_calculate_expression_division_by_zero_is_nan(mod):
    assert mod.calculate_expression("1 / 0") == "nan"


def test_calculate_expression_large_power_is_capped_to_inf(mod):
    assert mod.calculate_expression("2 ** 500") == "inf"


def test_calculate_expression_rejects_unknown_names(mod):
    result = mod.calculate_expression("foo + 1")
    assert result.startswith("Error:")


def test_calculate_expression_blocks_function_calls(mod):
    # A call node is not an allowed construct -> must fail closed.
    result = mod.calculate_expression("__import__('os')")
    assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# binary_search_solver
# ---------------------------------------------------------------------------

def test_binary_search_solver_converges_within_range(mod):
    out = mod.binary_search_solver(target=5.0, low=0.0, high=10.0)
    assert "mid-point calculated at" in out
    assert "steps" in out


def test_binary_search_solver_returns_string(mod):
    assert isinstance(mod.binary_search_solver(1.0, 0.0, 2.0), str)


# ---------------------------------------------------------------------------
# verify_ast_safety_and_structure
# ---------------------------------------------------------------------------

def test_ast_safety_accepts_plain_arithmetic(mod):
    ok, msg = mod.verify_ast_safety_and_structure("x = 1 + 2\ny = x * 3")
    assert ok is True
    assert "safe" in msg.lower()


def test_ast_safety_rejects_syntax_error(mod):
    ok, msg = mod.verify_ast_safety_and_structure("def broken(:\n    pass")
    assert ok is False
    assert "Syntax Error" in msg


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from socket import socket",
])
def test_ast_safety_blocks_prohibited_imports(mod, code):
    ok, msg = mod.verify_ast_safety_and_structure(code)
    assert ok is False
    assert "restricted module" in msg


@pytest.mark.parametrize("code", [
    "eval('1 + 1')",
    "exec('x = 1')",
    "getattr(object, 'foo')",
])
def test_ast_safety_blocks_dynamic_functions(mod, code):
    ok, msg = mod.verify_ast_safety_and_structure(code)
    assert ok is False
    assert "dynamic function" in msg


def test_ast_safety_blocks_dunder_string_literals(mod):
    ok, msg = mod.verify_ast_safety_and_structure("x = '__class__'")
    assert ok is False
    assert "double underscores" in msg


def test_ast_safety_blocks_hidden_attribute_access(mod):
    ok, msg = mod.verify_ast_safety_and_structure("obj.__dict__")
    assert ok is False
    assert "hidden attribute" in msg


# ---------------------------------------------------------------------------
# safe_literal_dict_eval
# ---------------------------------------------------------------------------

def _literal(mod, source):
    tree = ast.parse(source, mode="eval")
    return mod.safe_literal_dict_eval(tree)


def test_safe_literal_dict_eval_handles_containers(mod):
    assert _literal(mod, "{'a': 1, 'b': [1, 2, 3]}") == {"a": 1, "b": [1, 2, 3]}
    assert _literal(mod, "(1, 2, 3)") == (1, 2, 3)
    assert _literal(mod, "[True, False, None]") == [True, False, None]


def test_safe_literal_dict_eval_handles_unary(mod):
    assert _literal(mod, "-7") == -7
    assert _literal(mod, "+7") == 7


def test_safe_literal_dict_eval_rejects_non_literal(mod):
    with pytest.raises(ValueError):
        _literal(mod, "1 + 2")


# ---------------------------------------------------------------------------
# extract_json_block / soft_json_parse
# ---------------------------------------------------------------------------

def test_extract_json_block_after_prediction_marker(mod):
    text = 'Some reasoning...\nPrediction: {"answer": 42} trailing junk'
    assert mod.extract_json_block(text) == '{"answer": 42}'


def test_extract_json_block_handles_nested_braces(mod):
    text = 'Prediction: {"outer": {"inner": 1}} extra'
    assert mod.extract_json_block(text) == '{"outer": {"inner": 1}}'


def test_extract_json_block_without_braces_returns_empty(mod):
    assert mod.extract_json_block("no json here") == ""


def test_soft_json_parse_valid_json(mod):
    assert mod.soft_json_parse('Prediction: {"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_soft_json_parse_strips_trailing_commas(mod):
    assert mod.soft_json_parse('Prediction: {"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_soft_json_parse_falls_back_to_python_literals(mod):
    # Single-quoted keys are invalid JSON but valid python literals.
    assert mod.soft_json_parse("Prediction: {'a': 1}") == {"a": 1}


def test_soft_json_parse_returns_empty_on_garbage(mod):
    assert mod.soft_json_parse("totally not json") == {}


# ---------------------------------------------------------------------------
# secure_import
# ---------------------------------------------------------------------------

def test_secure_import_allows_math_and_json(mod):
    assert mod.secure_import("math") is math
    import json as _json
    assert mod.secure_import("json") is _json


def test_secure_import_blocks_everything_else(mod):
    with pytest.raises(ImportError):
        mod.secure_import("os")
