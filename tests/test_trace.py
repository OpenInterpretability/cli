"""Tests for openinterp.trace — smoke only. Full integration tests require [full] extras and GPU."""
import pytest
from openinterp import Trace, TraceFeature, generate_trace, TraceUnavailable


def test_trace_model_roundtrip():
    t = Trace(
        prompt="hi",
        model="m",
        layer="L0",
        sae_repo="user/repo",
        tokens=[" hi", " world"],
        features=[TraceFeature(id="f0", name="demo", desc="demo feat", auroc=0.5)],
        activations=[[0.1, 0.9]],
    )
    data = t.model_dump_json()
    assert "\"f0\"" in data
    assert "\"activations\":[[0.1,0.9]]" in data


def test_generate_trace_without_full_deps_raises():
    """If torch/transformers isn't installed, raise TraceUnavailable with a helpful message."""
    try:
        import torch  # noqa: F401
        # If torch is installed in test env, this test is N/A
        pytest.skip("torch installed — can't test the missing-deps path")
    except ImportError:
        pass
    with pytest.raises(TraceUnavailable) as exc:
        generate_trace(
            model_id="google/gemma-2-2b",
            prompt="hello",
            sae_repo="dummy/sae",
            layer=12,
        )
    assert "openinterp[full]" in str(exc.value)
