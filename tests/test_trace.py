import pytest
from openinterp import generate_trace


def test_generate_trace_raises_helpful_error():
    with pytest.raises(NotImplementedError) as exc:
        generate_trace("Qwen/Qwen3.6-27B", "hello")
    assert "Q2 2026" in str(exc.value)
    assert "openinterp.org" in str(exc.value)
