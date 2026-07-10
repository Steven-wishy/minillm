"""Tests for helpers that differ between pipeline versions, plus the curriculum
compiler / data ingestor that all versions share."""

import importlib
import json
import math

import pytest

V1_V2 = ["finetuning", "finetuning_v2"]
ALL = ["finetuning", "finetuning_v2", "finetuning_v3"]


class _DummySearchAdapter:
    def query(self, search_query):  # pragma: no cover - not triggered
        return "stub"


# ---------------------------------------------------------------------------
# inject_synthetic_bug  (V1 / V2 only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", V1_V2)
def test_inject_synthetic_bug_name_error_branch(name, monkeypatch):
    mod = importlib.import_module(name)
    monkeypatch.setattr(mod.random, "choice", lambda seq: seq[0])
    buggy, error = mod.inject_synthetic_bug("value = 1\nprint(value)", ["value"])
    assert "value_typo" in buggy
    assert "NameError" in error
    assert "value_typo" in error


@pytest.mark.parametrize("name", V1_V2)
def test_inject_synthetic_bug_import_error_branch(name):
    mod = importlib.import_module(name)
    # No variables -> deterministic fall through to the import-omission branch.
    buggy, error = mod.inject_synthetic_bug("import math\nr = math.sqrt(4)", [])
    assert "# import math omitted" in buggy
    assert "math_module." in buggy
    assert "math_module" in error


# ---------------------------------------------------------------------------
# has_consecutive_repetition_loops  (V1 / V2 only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", V1_V2)
def test_repetition_detector_short_text(name):
    mod = importlib.import_module(name)
    assert mod.has_consecutive_repetition_loops("abc", chunk_size=15) is False


@pytest.mark.parametrize("name", V1_V2)
def test_repetition_detector_detects_duplicate_chunk(name):
    mod = importlib.import_module(name)
    assert mod.has_consecutive_repetition_loops("abcabc", chunk_size=3) is True


@pytest.mark.parametrize("name", V1_V2)
def test_repetition_detector_no_repetition(name):
    mod = importlib.import_module(name)
    assert mod.has_consecutive_repetition_loops("abcdefghij", chunk_size=3) is False


# ---------------------------------------------------------------------------
# evaluate_batch  (V1 / V2 only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", V1_V2)
def test_evaluate_batch_maps_each_task(name):
    mod = importlib.import_module(name)
    sandbox = mod.TeacherSandbox(_DummySearchAdapter())
    tasks = [
        ("a = 1", "assert a == 1", ["a"]),
        ("b = 2", "assert b == 999", ["b"]),
    ]
    results = sandbox.evaluate_batch(tasks)
    assert len(results) == 2
    assert results[0][0] is True
    assert results[1][0] is False


# ---------------------------------------------------------------------------
# validate_layout_xml_structure  (V3 only)
# ---------------------------------------------------------------------------

def test_validate_layout_xml_balanced():
    mod = importlib.import_module("finetuning_v3")
    ok, msg = mod.validate_layout_xml_structure("<think><dream_code></dream_code></think>")
    assert ok is True
    assert "validated" in msg.lower()


def test_validate_layout_xml_mismatched():
    mod = importlib.import_module("finetuning_v3")
    ok, msg = mod.validate_layout_xml_structure("<think></dream_code>")
    assert ok is False
    assert "Mismatched" in msg


def test_validate_layout_xml_unclosed():
    mod = importlib.import_module("finetuning_v3")
    ok, msg = mod.validate_layout_xml_structure("<think><critique></critique>")
    assert ok is False
    assert "Unclosed" in msg


# ---------------------------------------------------------------------------
# compute_dense_reward  (V3 only)
# ---------------------------------------------------------------------------

def test_compute_dense_reward_full_credit():
    mod = importlib.import_module("finetuning_v3")
    # A single-word turn contributes log(1) == 0 length penalty.
    reward = mod.compute_dense_reward("done", passed=True, xml_valid=True, has_critique=True)
    assert reward == pytest.approx(1.5)


def test_compute_dense_reward_applies_length_penalty():
    mod = importlib.import_module("finetuning_v3")
    reward = mod.compute_dense_reward("one two three", passed=False, xml_valid=False, has_critique=False)
    assert reward == pytest.approx(-0.05 * math.log(3))


def test_compute_dense_reward_partial_components():
    mod = importlib.import_module("finetuning_v3")
    reward = mod.compute_dense_reward("word", passed=True, xml_valid=False, has_critique=True)
    assert reward == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# ProceduralCurriculumCompiler / DynamicDataIngestor  (all versions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_generate_curriculum_produces_well_formed_entries(name):
    mod = importlib.import_module(name)
    sandbox = mod.TeacherSandbox(_DummySearchAdapter())
    compiler = mod.ProceduralCurriculumCompiler(mod.MULTI_DOMAIN_SEED_CORPUS, sandbox)
    curriculum = compiler.generate_curriculum(size=4)
    assert len(curriculum) == 4
    required_keys = {"domain", "question", "variables", "sft_reference", "test_assertion"}
    for item in curriculum:
        assert required_keys.issubset(item.keys())
        assert isinstance(item["variables"], list)
        assert "<dream_code>" in item["sft_reference"]


def test_v3_data_ingestor_loads_custom_dataset(tmp_path):
    mod = importlib.import_module("finetuning_v3")
    sandbox = mod.TeacherSandbox(_DummySearchAdapter())
    compiler = mod.ProceduralCurriculumCompiler(mod.MULTI_DOMAIN_SEED_CORPUS, sandbox)
    ingestor = mod.DynamicDataIngestor(compiler, sandbox)

    dataset = [{"question": "q", "sft_reference": "r", "test_assertion": "assert True", "variables": []}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    loaded = ingestor.load_training_dataset(str(path))
    assert loaded == dataset


def test_v3_data_ingestor_falls_back_to_curriculum(tmp_path):
    mod = importlib.import_module("finetuning_v3")
    sandbox = mod.TeacherSandbox(_DummySearchAdapter())
    compiler = mod.ProceduralCurriculumCompiler(mod.MULTI_DOMAIN_SEED_CORPUS, sandbox)
    ingestor = mod.DynamicDataIngestor(compiler, sandbox)

    loaded = ingestor.load_training_dataset(str(tmp_path / "missing.json"))
    assert isinstance(loaded, list)
    assert len(loaded) > 0
