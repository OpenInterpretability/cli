"""Atlas — cross-model feature search."""
from typing import Optional
import requests
from openinterp.models import AtlasFeature

API_BASE = "https://openinterp.org/api"


def search_features(
    query: str,
    model: Optional[str] = None,
    limit: int = 20,
    timeout: int = 15,
) -> list[AtlasFeature]:
    """
    Semantic search across the Atlas of SAE features.

    Args:
        query: natural language query, e.g. "overconfidence" or "medical reasoning"
        model: optional HF model ID filter, e.g. "Qwen/Qwen3.6-27B"
        limit: max results
        timeout: HTTP timeout seconds

    Returns:
        list[AtlasFeature]

    Notes:
        Atlas is scheduled for Q2 2026 launch. Until then this function returns
        a curated subset of features from the currently-shipped Trace Theater
        demo on openinterp.org (so the SDK is functional even before the Atlas
        backend lands). See https://openinterp.org/observatory/atlas
    """
    try:
        r = requests.get(
            f"{API_BASE}/atlas/search",
            params={"q": query, "model": model, "limit": limit},
            timeout=timeout,
        )
        if r.status_code == 200:
            return [AtlasFeature(**item) for item in r.json().get("results", [])]
    except requests.RequestException:
        pass

    # Fallback: curated features from the public Trace Theater demo.
    return _curated_fallback(query, limit)


def get_feature(feature_id: str, model: str) -> Optional[AtlasFeature]:
    """Fetch a single feature by (feature_id, model)."""
    try:
        r = requests.get(f"{API_BASE}/atlas/feature", params={"id": feature_id, "model": model}, timeout=15)
        if r.status_code == 200:
            return AtlasFeature(**r.json())
    except requests.RequestException:
        pass
    return None


def _curated_fallback(query: str, limit: int) -> list[AtlasFeature]:
    """Offline fallback — matches against shipped demo features."""
    curated = [
        AtlasFeature(id="f2503", name="overconfidence_pattern",
                     description="Definitive clinical commitments without hedging qualifiers.",
                     model="Qwen/Qwen3.6-27B", layer="L31 residual",
                     sae_repo="caiovicentino1/qwen36-27b-sae-multilayer", auroc=0.54),
        AtlasFeature(id="f3383", name="medical_domain_terms",
                     description="Medical terminology activation (syndrome, coronary, aspirin).",
                     model="Qwen/Qwen3.6-27B", layer="L31 residual",
                     sae_repo="caiovicentino1/qwen36-27b-sae-multilayer", auroc=0.72),
        AtlasFeature(id="f1847", name="urgency_assessment",
                     description="Time-critical decision signal; peaks on imperative verbs.",
                     model="Qwen/Qwen3.6-27B", layer="L31 residual",
                     sae_repo="caiovicentino1/qwen36-27b-sae-multilayer", auroc=0.68),
        AtlasFeature(id="f567", name="hedging_language",
                     description="\"may\", \"might\", \"possibly\" — healthy calibration signal.",
                     model="Qwen/Qwen3.6-27B", layer="L31 residual",
                     sae_repo="caiovicentino1/qwen36-27b-sae-multilayer", auroc=0.58),
    ]
    q = query.lower()
    matches = [f for f in curated if q in f.name.lower() or q in f.description.lower()]
    if not matches:
        matches = curated
    return matches[:limit]
