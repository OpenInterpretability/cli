from pydantic import BaseModel, Field
from typing import Optional


class AtlasFeature(BaseModel):
    id: str                                    # e.g. "f2503"
    name: str                                  # e.g. "overconfidence_pattern"
    description: str
    model: str                                 # e.g. "Qwen/Qwen3.6-27B"
    layer: str                                 # e.g. "L31 residual"
    sae_repo: str
    auroc: Optional[float] = None
    top_activating_tokens: list[str] = Field(default_factory=list)


class Trace(BaseModel):
    id: str
    prompt: str
    model: str
    tokens: list[str]
    feature_ids: list[str]
    activations: list[list[float]]             # [features][tokens]
    created_at: Optional[str] = None
    url: Optional[str] = None                  # shareable openinterp.org URL
