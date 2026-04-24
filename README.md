<div align="center">

# `openinterp`

### Python SDK + CLI for [openinterp.org](https://openinterp.org)

Search the feature Atlas, generate Traces from your own SAE, rank against the public InterpScore leaderboard.

[![PyPI](https://img.shields.io/pypi/v/openinterp.svg?color=8b5cf6)](https://pypi.org/project/openinterp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![openinterp.org](https://img.shields.io/badge/site-openinterp.org-8b5cf6)](https://openinterp.org)
[![Discussions](https://img.shields.io/github/discussions/OpenInterpretability/cli)](https://github.com/OpenInterpretability/cli/discussions)

</div>

---

## Install

```bash
pip install openinterp              # lite: Atlas + CLI (no torch, ~2 MB total)
pip install "openinterp[full]"      # + torch/transformers/safetensors for trace generation
```

Requires **Python ≥ 3.10**.

---

## Part of a 5-repo ecosystem

| Repo | What's in it |
|---|---|
| [`.github`](https://github.com/OpenInterpretability/.github) | Org profile + shared CoC + SECURITY |
| [`web`](https://github.com/OpenInterpretability/web) | Next.js site behind openinterp.org |
| [`notebooks`](https://github.com/OpenInterpretability/notebooks) | 23 training + interpretability notebooks |
| **`cli`** (you are here) | `pip install openinterp` — Python SDK |
| [`mechreward`](https://github.com/OpenInterpretability/mechreward) | SAE features as dense RL reward |

---

## 🚀 Quick start

### Search the Atlas (offline, zero GPU)

```bash
$ openinterp atlas "overconfidence"
```

```
                    Atlas results: 'overconfidence'
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━
┃ ID      ┃ Name                    ┃ Model             ┃ AUROC ┃ Description
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━
│ f2503   │ overconfidence_pattern  │ Qwen/Qwen3.6-27B  │  0.54 │ Definitive…
│ f1847   │ urgency_assessment      │ Qwen/Qwen3.6-27B  │  0.68 │ Time-critic…
└─────────┴─────────────────────────┴───────────────────┴───────┴────────────
```

```python
>>> from openinterp import search_features
>>> features = search_features("overconfidence", model="Qwen/Qwen3.6-27B")
>>> features[0].id
'f2503'
```

### Generate a Trace from your own SAE

```bash
pip install "openinterp[full]"

openinterp trace \
    --model google/gemma-2-2b \
    --sae-repo YOUR_HF_USER/gemma2-2b-sae-first \
    --prompt "The capital of France is" \
    --layer 12 \
    --d-model 2304 --d-sae 16384 --k 64 \
    --out my_trace.json
```

This:
1. Loads the base model in bf16 with SDPA (no flash-attn)
2. Loads your SAE from HuggingFace (sae_lens `safetensors` format)
3. Generates tokens, captures residuals at layer 12
4. Applies the SAE, picks top-10 active features
5. Writes a `Trace` JSON matching [openinterp.org/observatory/trace](https://openinterp.org/observatory/trace) byte-for-byte

### Python API

```python
from openinterp import generate_trace

trace = generate_trace(
    model_id="google/gemma-2-2b",
    sae_repo="YOUR_HF_USER/gemma2-2b-sae-first",
    prompt="The capital of France is",
    layer=12,
    d_model=2304,
    d_sae=16384,
    k=64,
)

print(trace.model_dump_json(indent=2))   # Trace Theater schema
```

### With feature labels from notebook 04

```bash
# After running 04_discover_features.ipynb (emits feature_catalog.json):
openinterp trace ... --catalog feature_catalog.json
```

Trace features inherit names from your catalog.

---

## 📦 What's in v0.1.0

| Command | Status | What it does |
|---|---|---|
| `openinterp atlas <query>` | ✅ live | Feature search with offline fallback to curated demo features |
| `openinterp trace ...` | ✅ live (needs `[full]`) | Real SAE trace generation, sae_lens format, any HF model |
| `openinterp info` | ✅ live | Version + optional-dep status |

### Planned v0.2.0 (Q2 2026)

- `openinterp upload-trace <trace.json>` → shareable openinterp.org URL
- `openinterp score --sae-repo X` → compute InterpScore (wraps [notebook 18](https://github.com/OpenInterpretability/notebooks/blob/main/notebooks/18_interpscore_eval.ipynb))
- `openinterp steer --sae-repo X --feature Y --alpha Z` → intervention (wraps [notebook 06](https://github.com/OpenInterpretability/notebooks/blob/main/notebooks/06_steer_your_model.ipynb))
- `openinterp circuit --sae-repo X --prompt Y` → attribution graph JSON (wraps [notebook 14/15](https://github.com/OpenInterpretability/notebooks/))
- `openinterp publish <repo>` → HuggingFace release with model card

Open an issue on the [tracker](https://github.com/OpenInterpretability/cli/issues) if you'd take one of these.

---

## 🛠️ Development

```bash
git clone https://github.com/OpenInterpretability/cli openinterp-cli
cd openinterp-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"          # dev = pytest + ruff + build; full = torch + transformers
pytest -xvs                            # 5 tests, ~1s
```

### Package layout

```
openinterp-cli/
├── pyproject.toml              # name='openinterp', hatchling build
├── openinterp/
│   ├── __init__.py             # public exports + __version__
│   ├── models.py               # pydantic types: AtlasFeature, Trace, TraceFeature
│   ├── atlas.py                # search_features() — HF API + curated fallback
│   ├── trace.py                # generate_trace() — real transformers-based impl
│   └── cli.py                  # click-based CLI: atlas / trace / info
├── tests/
│   ├── test_atlas.py
│   └── test_trace.py
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

### Contribution recipe — add a new command

> Full rules: [CONTRIBUTING.md](./CONTRIBUTING.md).

1. Decide which notebook it wraps (score → 18, steer → 06, circuit → 14/15, publish → generic)
2. Add a function to the matching file (`openinterp/score.py`, etc.). Keep it small — actual compute lives in the notebook.
3. Expose it in `__init__.py`
4. Add a `@main.command()` in `cli.py` with click decorators
5. Add a smoke test in `tests/test_<name>.py`
6. Update `CHANGELOG.md` under `[Unreleased]`
7. PR title: `Add openinterp <command>`

**Hard rules**:
- Python ≥ 3.10 syntax (PEP 604 unions OK)
- `dtype=torch.bfloat16`, never `torch_dtype=` (transformers 5.x deprecated)
- SDPA only, never flash-attn
- New heavy deps (`torch` tier) → add to `[full]` extra, not base
- Every new public function has type hints + docstring

---

## 🚢 Release process (maintainer)

```bash
# 1. Bump version in BOTH:
#    pyproject.toml          ([project] version = "X.Y.Z")
#    openinterp/__init__.py  (__version__ = "X.Y.Z")
# 2. Update CHANGELOG.md — move [Unreleased] → [X.Y.Z] — YYYY-MM-DD

source .venv/bin/activate
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload dist/*     # needs PyPI token in ~/.pypirc

git tag vX.Y.Z
git push --tags
```

---

## CI

Every PR runs:
- `pytest -xvs` across Python 3.10, 3.11, 3.12 (see `.github/workflows/ci.yml`)
- `ruff check .` (warn-only for now)
- `python -m build` + `twine check`

Green required to merge.

---

## Community

- 💬 [Discussions](https://github.com/OpenInterpretability/cli/discussions) — API proposals, "which repo should this live in"
- 🟢 [Good-first-issues](https://github.com/OpenInterpretability/cli/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
- 📦 [PyPI release history](https://pypi.org/project/openinterp/#history)
- ✉️ hi@openinterp.org

---

## Standing on the shoulders of

- [Neuronpedia](https://neuronpedia.org) · the SAE encyclopedia
- [Gemma Scope](https://huggingface.co/google/gemma-scope) · reference SAE suite
- [Gao et al. 2024](https://arxiv.org/abs/2406.04093) · TopK + AuxK recipe
- [SAELens](https://github.com/jbloomAus/SAELens) · our safetensors format

---

## License

**MIT.** Built by [Caio Vicentino](https://huggingface.co/caiovicentino1) + OpenInterpretability. 2026.

[openinterp.org](https://openinterp.org) · [github.com/OpenInterpretability](https://github.com/OpenInterpretability) · [hi@openinterp.org](mailto:hi@openinterp.org)
