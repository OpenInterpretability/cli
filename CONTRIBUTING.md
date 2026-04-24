# Contributing to `openinterp` (Python SDK + CLI)

Thanks for helping ship the package. This repo is the codebase behind `pip install openinterp`.

## Setup

```bash
git clone https://github.com/OpenInterpretability/cli openinterp-cli
cd openinterp-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"          # full = torch+transformers, dev = pytest+ruff+build
pytest -xvs
```

## Scope

| In scope | Out of scope |
|---|---|
| New commands wrapping the notebooks (`score`, `steer`, `circuit`, `publish`) | Training SAEs (→ `notebooks`) |
| Atlas API integration when openinterp.org/api ships (Q2) | UI components (→ `web`) |
| Adapter integrations (SAELens, TransformerLens, nnsight, sae-bench) | RL reward research (→ `mechreward`) |
| Performance (bf16 paths, `torch.compile` wins, `accelerate` multi-GPU) | |
| Type hints, docs, tests, error messages | |

## Layout

```
openinterp/
├── __init__.py     # public exports + __version__
├── atlas.py        # feature search (offline fallback + future HF API)
├── trace.py        # trace generation (needs [full] extras)
├── models.py       # pydantic types
└── cli.py          # click-based CLI
tests/
├── test_atlas.py
└── test_trace.py
```

## Coding conventions

- Python ≥ 3.10 syntax (PEP 604 unions are fine).
- `ruff check .` + `ruff format .` before pushing.
- Type-hint every public function.
- Keep optional deps truly optional — wrap imports inside functions, raise `TraceUnavailable` with an install hint.
- Never use `torch_dtype=` (transformers deprecated it); use `dtype=`.
- Never pin `flash-attn` — we use SDPA across the board.

## Releasing (maintainer-only)

```bash
# Bump __version__ in __init__.py AND pyproject.toml.
# Update CHANGELOG.md (move [Unreleased] → [x.y.z] — YYYY-MM-DD).
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload dist/*
git tag vX.Y.Z && git push --tags
```

## Questions

[Open a Discussion](https://github.com/OpenInterpretability/cli/discussions) or DM @openinterp on X.
