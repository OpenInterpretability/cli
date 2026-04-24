from openinterp import search_features, AtlasFeature


def test_search_returns_features():
    results = search_features("overconfidence", limit=5)
    assert isinstance(results, list)
    assert all(isinstance(f, AtlasFeature) for f in results)


def test_search_matches_curated():
    results = search_features("overconfidence", limit=10)
    ids = [f.id for f in results]
    assert "f2503" in ids


def test_search_returns_fallback_on_empty():
    results = search_features("no_such_feature_xyz_123", limit=5)
    assert len(results) > 0  # fallback returns curated anyway
