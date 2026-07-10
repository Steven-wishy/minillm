"""Shared pytest fixtures and lightweight stubs.

The finetuning pipeline modules import a couple of heavy, GPU-oriented
third-party packages (``unsloth`` and ``peft``) unconditionally at import
time.  Those packages are not required for the pure-Python helper logic that
the unit tests exercise, so we install minimal stand-ins in ``sys.modules``
before the modules under test are imported.  This keeps the test suite fast
and runnable on a plain CPU machine without pulling in the full training
stack.
"""

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_stub(name, attrs):
    """Register a stub module under ``name`` if it is not already importable."""
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module


class _StubFastLanguageModel:
    """No-op stand-in for ``unsloth.FastLanguageModel``."""

    @classmethod
    def for_inference(cls, *args, **kwargs):
        return None

    @classmethod
    def for_training(cls, *args, **kwargs):
        return None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - unused
        raise RuntimeError("Model loading is not available in the test stub.")

    @classmethod
    def get_peft_model(cls, *args, **kwargs):  # pragma: no cover - unused
        raise RuntimeError("PEFT wrapping is not available in the test stub.")


def _patch_fast_rl(*args, **kwargs):
    return None


def _get_peft_model_state_dict(*args, **kwargs):
    return {}


_install_stub(
    "unsloth",
    {"FastLanguageModel": _StubFastLanguageModel, "PatchFastRL": _patch_fast_rl},
)
_install_stub(
    "peft",
    {
        "get_peft_model_state_dict": _get_peft_model_state_dict,
        "LoraConfig": object,
        "get_peft_model": lambda *a, **k: None,
    },
)
