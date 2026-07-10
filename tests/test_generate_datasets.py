"""Tests for the dataset synthesis utilities.

Only the pure, side-effect-free helpers are exercised here (the file-writing /
network driver lives behind a ``__main__`` guard).
"""

import importlib
import json

import pytest

MODULE_NAMES = ["generate_datasets", "generate_datasets_v2"]

REQUIRED_TEMPLATE_KEYS = {
    "domain",
    "question",
    "variables",
    "sft_reference",
    "test_assertion",
    "system_prompt",
}


@pytest.fixture(params=MODULE_NAMES)
def mod(request):
    return importlib.import_module(request.param)


def _template_functions(mod):
    return [
        getattr(mod, name)
        for name in dir(mod)
        if name.startswith("get_") and name.endswith("_template")
    ]


def test_module_exposes_template_functions(mod):
    assert len(_template_functions(mod)) == 15


def test_templates_have_required_shape(mod):
    for func in _template_functions(mod):
        item = func(0)
        assert isinstance(item, dict), func.__name__
        assert REQUIRED_TEMPLATE_KEYS.issubset(item.keys()), func.__name__
        assert isinstance(item["variables"], list)
        # Every template must be JSON serialisable since it is written to disk.
        json.dumps(item)


def test_templates_vary_with_index(mod):
    for func in _template_functions(mod):
        assert func(0)["question"] != func(1)["question"], func.__name__


# ---------------------------------------------------------------------------
# deduplicate_dataset
# ---------------------------------------------------------------------------

def test_deduplicate_removes_exact_duplicates(mod):
    dataset = [
        {"question": "solve x"},
        {"question": "solve x"},
        {"question": "solve y"},
    ]
    result = mod.deduplicate_dataset(dataset)
    assert [d["question"] for d in result] == ["solve x", "solve y"]


def test_deduplicate_normalises_whitespace(mod):
    dataset = [
        {"question": "hello   world"},
        {"question": "hello world"},
        {"question": " hello world "},
    ]
    assert len(mod.deduplicate_dataset(dataset)) == 1


def test_deduplicate_handles_missing_question_key(mod):
    dataset = [{"foo": 1}, {"foo": 2}]
    # Both normalise to an empty prompt -> collapse to a single entry.
    assert len(mod.deduplicate_dataset(dataset)) == 1


def test_deduplicate_preserves_order(mod):
    dataset = [
        {"question": "c"},
        {"question": "a"},
        {"question": "b"},
        {"question": "a"},
    ]
    assert [d["question"] for d in mod.deduplicate_dataset(dataset)] == ["c", "a", "b"]


def test_deduplicate_empty_input(mod):
    assert mod.deduplicate_dataset([]) == []
