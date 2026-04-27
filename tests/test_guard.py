"""Tests for openinterp.guard — smoke + structure only.

Full integration tests require [full] extras + GPU + downloading the probe
artifact, which is too heavy for CI. We validate API surface here and let
the notebook (notebooks/30, notebooks/31) be the integration test.
"""
import pytest

from openinterp import (
    FabricationGuard,
    FabricationGuardError,
    GuardOutput,
)


# ----------------------------------------------------- structure / public API

def test_class_is_exported():
    assert FabricationGuard is not None
    assert FabricationGuardError is not None
    assert GuardOutput is not None


def test_default_registry_has_qwen36_27b():
    from openinterp.guard import DEFAULT_PROBE_REGISTRY
    assert "Qwen/Qwen3.6-27B" in DEFAULT_PROBE_REGISTRY
    repo = DEFAULT_PROBE_REGISTRY["Qwen/Qwen3.6-27B"]
    assert "FabricationGuard" in repo or "fabrication" in repo.lower()


def test_unknown_model_id_raises():
    with pytest.raises(FabricationGuardError):
        FabricationGuard.from_pretrained("nonexistent/model-xyz")


def test_invalid_mode_raises():
    # Construct without attach to test mode validation
    g = FabricationGuard(probe=None, scaler=None, layer=31, threshold=0.5)
    with pytest.raises(FabricationGuardError):
        g.generate("hi", mode="invalid_mode")


def test_repr_format():
    g = FabricationGuard(probe=None, scaler=None, layer=31, threshold=0.684)
    s = repr(g)
    assert "FabricationGuard" in s
    assert "31" in s
    assert "0.684" in s
    assert "detached" in s


# --------------------------------------------------------------- GuardOutput

def test_guard_output_as_dict():
    out = GuardOutput(
        text="hi", score=0.42, flagged=False,
        mode="detect", abstained=False, threshold=0.7,
    )
    d = out.as_dict()
    assert d["text"] == "hi"
    assert d["score"] == 0.42
    assert d["flagged"] is False
    assert d["mode"] == "detect"
    assert d["abstained"] is False
    assert d["threshold"] == 0.7


# --------------------------------------------------- attach without dependency

def test_attach_requires_full():
    """If torch/transformers/joblib/sklearn are missing, score() must raise."""
    g = FabricationGuard(probe=None, scaler=None, layer=31, threshold=0.5)
    with pytest.raises(FabricationGuardError):
        g.score("test")


# ------------------------------------------------- context manager / cleanup

def test_context_manager():
    g = FabricationGuard(probe=None, scaler=None, layer=31, threshold=0.5)
    with g as inner:
        assert inner is g
    # close was called on exit; calling close again should be safe
    g.close()
    g.close()


def test_threshold_default():
    g = FabricationGuard(probe=None, scaler=None, layer=31, threshold=0.684)
    assert g.threshold == 0.684


def test_abstain_response_override():
    g = FabricationGuard(
        probe=None, scaler=None, layer=31, threshold=0.5,
        abstain_response="custom uncertainty",
    )
    assert g.abstain_response == "custom uncertainty"
