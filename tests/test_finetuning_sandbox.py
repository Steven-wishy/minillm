"""Tests for the AST loop-limiter, the in-process teacher sandbox, the SLERP
adapter merge, and the search adapter cache."""

import ast
import importlib

import pytest
import torch

MODULE_NAMES = ["finetuning", "finetuning_v2", "finetuning_v3"]


@pytest.fixture(params=MODULE_NAMES)
def mod(request):
    return importlib.import_module(request.param)


class _DummySearchAdapter:
    def query(self, search_query):  # pragma: no cover - not triggered
        return "stub"


def _make_sandbox(mod, loop_limit=5000):
    return mod.TeacherSandbox(_DummySearchAdapter(), loop_limit=loop_limit)


# ---------------------------------------------------------------------------
# LoopLimiterTransformer
# ---------------------------------------------------------------------------

def _run_transformed(mod, source, limit):
    tree = ast.parse(source)
    transformed = mod.LoopLimiterTransformer(limit=limit).visit(tree)
    ast.fix_missing_locations(transformed)
    code = compile(transformed, "<test>", "exec")
    namespace = {}
    exec(code, namespace)
    return namespace


def test_loop_limiter_allows_bounded_loops(mod):
    ns = _run_transformed(mod, "total = 0\nfor i in range(5):\n    total += i", limit=100)
    assert ns["total"] == 10


def test_loop_limiter_aborts_infinite_while(mod):
    with pytest.raises(TimeoutError):
        _run_transformed(mod, "x = 0\nwhile True:\n    x += 1", limit=25)


def test_loop_limiter_aborts_runaway_for(mod):
    with pytest.raises(TimeoutError):
        _run_transformed(mod, "acc = 0\nfor i in range(10_000):\n    acc += 1", limit=50)


# ---------------------------------------------------------------------------
# TeacherSandbox.execute_safely
# ---------------------------------------------------------------------------

def test_sandbox_executes_and_extracts_variables(mod):
    sandbox = _make_sandbox(mod)
    passed, variables, log = sandbox.execute_safely("x = 2 + 3", "assert x == 5", ["x"])
    assert passed is True
    assert variables == {"x": 5}
    assert "successful" in log.lower()


def test_sandbox_reports_failed_assertions(mod):
    sandbox = _make_sandbox(mod)
    passed, variables, log = sandbox.execute_safely("x = 1", "assert x == 999", ["x"])
    assert passed is False
    assert variables == {}
    assert "assertion" in log.lower()


def test_sandbox_blocks_unsafe_code(mod):
    sandbox = _make_sandbox(mod)
    passed, variables, log = sandbox.execute_safely("import os", "assert True", [])
    assert passed is False
    assert "Security Block" in log


def test_sandbox_enforces_loop_limit(mod):
    sandbox = _make_sandbox(mod, loop_limit=25)
    passed, _, log = sandbox.execute_safely("x = 0\nwhile True:\n    x += 1", "assert False", [])
    assert passed is False
    assert "Loop limit" in log


def test_sandbox_caches_results(mod):
    sandbox = _make_sandbox(mod)
    first = sandbox.execute_safely("y = 7", "assert y == 7", ["y"])
    assert len(sandbox.cache) == 1
    second = sandbox.execute_safely("y = 7", "assert y == 7", ["y"])
    assert first == second
    assert len(sandbox.cache) == 1


def test_sandbox_captures_print_output(mod):
    sandbox = _make_sandbox(mod)
    passed, _, log = sandbox.execute_safely("print('hello sandbox')\nz = 1", "assert z == 1", ["z"])
    assert passed is True
    assert "hello sandbox" in log


# ---------------------------------------------------------------------------
# run_slerp_adapter_merge
# ---------------------------------------------------------------------------

def test_slerp_merge_identical_weights_is_stable(mod):
    weights = {"w": torch.tensor([3.0, 4.0])}
    merged = mod.run_slerp_adapter_merge(None, weights, {"w": torch.tensor([3.0, 4.0])}, t=0.5)
    assert torch.allclose(merged["w"], torch.tensor([3.0, 4.0]), atol=1e-4)


def test_slerp_merge_copies_keys_missing_in_second(mod):
    weights_v0 = {"only_v0": torch.tensor([1.0, 2.0])}
    merged = mod.run_slerp_adapter_merge(None, weights_v0, {}, t=0.5)
    assert torch.allclose(merged["only_v0"], torch.tensor([1.0, 2.0]))


def test_slerp_merge_zero_norm_uses_linear_interpolation(mod):
    v0 = {"w": torch.zeros(3)}
    v1 = {"w": torch.tensor([2.0, 2.0, 2.0])}
    merged = mod.run_slerp_adapter_merge(None, v0, v1, t=0.5)
    assert torch.allclose(merged["w"], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)


def test_slerp_merge_preserves_shape_and_dtype(mod):
    v0 = {"w": torch.randn(2, 3)}
    v1 = {"w": torch.randn(2, 3)}
    merged = mod.run_slerp_adapter_merge(None, v0, v1, t=0.3)
    assert merged["w"].shape == (2, 3)
    assert merged["w"].dtype == v0["w"].dtype


# ---------------------------------------------------------------------------
# RealSearchAdapter cache behaviour (no network)
# ---------------------------------------------------------------------------

def test_search_adapter_empty_query(mod, tmp_path):
    adapter = mod.RealSearchAdapter(cache_file=str(tmp_path / "cache.json"))
    assert adapter.query("   ") == "Error: Empty query parameter received."


def test_search_adapter_returns_cached_result(mod, tmp_path):
    adapter = mod.RealSearchAdapter(cache_file=str(tmp_path / "cache.json"))
    adapter.cache["python gil"] = "cached answer"
    assert adapter.query("python gil") == "cached answer"


def test_search_adapter_cache_round_trip(mod, tmp_path):
    cache_file = tmp_path / "cache.json"
    adapter = mod.RealSearchAdapter(cache_file=str(cache_file))
    adapter.cache["k"] = "v"
    adapter.save_cache()

    reloaded = mod.RealSearchAdapter(cache_file=str(cache_file))
    assert reloaded.cache == {"k": "v"}
