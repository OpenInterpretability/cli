"""Trace — generate per-token feature activations from model + prompt + SAE."""
from typing import Optional
from openinterp.models import Trace


def generate_trace(
    model_id: str,
    prompt: str,
    sae_repo: Optional[str] = None,
    layer: Optional[int] = None,
    max_new_tokens: int = 40,
) -> Trace:
    """
    Generate a feature-activation Trace for a prompt.

    Requires: transformers, safetensors, access to the model weights + SAE.

    Coming in v0.1.0 (Q2 2026). Until then use the public interactive
    Trace Theater at https://openinterp.org/observatory/trace
    """
    raise NotImplementedError(
        "generate_trace() is a v0.1.0 feature (Q2 2026). "
        "Use https://openinterp.org/observatory/trace for the public interactive trace, "
        "or follow the notebooks at https://github.com/OpenInterpretability/notebooks "
        "to compute activations yourself."
    )


def upload_trace(trace: Trace, public: bool = True) -> str:
    """
    Upload a Trace to openinterp.org; returns a shareable URL.

    Coming in v0.1.0 (Q2 2026).
    """
    raise NotImplementedError(
        "upload_trace() is a v0.1.0 feature. Coming Q2 2026."
    )
