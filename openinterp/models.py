"""Pydantic models shared across the SDK."""
from pydantic import BaseModel, Field
from typing import Optional


class AtlasFeature(BaseModel):
    """A feature entry in the cross-model Atlas."""

    id: str                                    # e.g. "f2503"
    name: str                                  # e.g. "overconfidence_pattern"
    description: str
    model: str                                 # e.g. "Qwen/Qwen3.6-27B"
    layer: str                                 # e.g. "L31 residual"
    sae_repo: str
    auroc: Optional[float] = None
    top_activating_tokens: list[str] = Field(default_factory=list)


class TraceFeature(BaseModel):
    """One feature inside a Trace — matches openinterp.org Trace Theater schema."""

    id: str
    name: str
    desc: str
    auroc: float = 0.0


class Trace(BaseModel):
    """
    A Trace: per-token feature activations from a model+SAE+prompt.

    Matches the openinterp.org Trace Theater JSON schema
    (lib/trace-data.ts) byte-compatible.
    """

    prompt: str
    model: str
    layer: str
    sae_repo: str
    tokens: list[str]
    features: list[TraceFeature]
    # activations[feature_idx][token_idx], values in [0, 1]
    activations: list[list[float]]
    counterfactuals: dict[str, dict[str, str]] = Field(default_factory=dict)
    # Optional metadata
    url: Optional[str] = None
    created_at: Optional[str] = None
