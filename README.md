# openinterp

> Python SDK + CLI for [openinterp.org](https://openinterp.org) — search the feature Atlas, generate Traces from your SAE, rank against the InterpScore leaderboard.

[![PyPI](https://img.shields.io/pypi/v/openinterp.svg)](https://pypi.org/project/openinterp/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![openinterp.org](https://img.shields.io/badge/site-openinterp.org-8b5cf6)](https://openinterp.org)

---

## Install

```bash
pip install openinterp              # lite: Atlas search + CLI (no torch)
pip install "openinterp[full]"      # + torch/transformers/safetensors for trace generation
```

Requires Python ≥ 3.10.

---

## Quick start

### Search the Atlas (offline, no GPU needed)

```bash
$ openinterp atlas "overconfidence"
```

```
        Atlas results: 'overconfidence'
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━…
┃ ID    ┃ Name                  ┃ Model            ┃ AUROC ┃ Description
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━…
│ f2503 │ overconfidence_patt…  │ Qwen/Qwen3.6-27B │  0.54 │ Definitive …
│ f1847 │ urgency_assessment    │ Qwen/Qwen3.6-27B │  0.68 │ Time-critic…
└───────┴───────────────────────┴──────────────────┴───────┴────────────…
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
1. Loads the base model in bf16 with SDPA
2. Loads your SAE from HuggingFace (sae_lens format)
3. Generates 30 tokens, captures residuals at layer 12
4. Applies the SAE, picks top-10 active features
5. Writes a `Trace` JSON that matches [openinterp.org/observatory/trace](https://openinterp.org/observatory/trace) exactly

### Use the Trace in the Python API

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

print(trace.model_dump_json(indent=2))  # Exact Trace Theater schema
```

### Optionally attach feature labels from notebook 04

```bash
# After running 04_discover_features.ipynb and saving feature_catalog.json:
openinterp trace ... --catalog feature_catalog.json
```

---

## What's in v0.1.0

| Command | Status | What it does |
|---|---|---|
| `openinterp atlas <query>` | ✅ Live | Feature search across the public Atlas, with offline fallback to the shipped demo features |
| `openinterp trace ...` | ✅ Live (needs `[full]`) | Real SAE-based trace generation, sae_lens format, any HF model |
| `openinterp info` | ✅ Live | Show version + optional dep status |

Planned for **v0.2.0 (Q2 2026)**:
- `openinterp upload-trace trace.json` → get a shareable openinterp.org URL
- `openinterp score --sae-repo X` → compute InterpScore locally (wraps notebook 18)
- `openinterp steer --sae-repo X --feature Y --alpha Z` → live intervention (wraps notebook 06)
- `openinterp circuit --sae-repo X --prompt Y` → attribution graph JSON (wraps notebook 14/15)
- `openinterp publish <repo> <artifact>` → HuggingFace release with model card

---

## Standing on the shoulders of

- [Neuronpedia](https://neuronpedia.org) — the SAE encyclopedia
- [Gemma Scope](https://huggingface.co/google/gemma-scope) — reference at-scale SAE suite
- [Gao et al. 2024](https://arxiv.org/abs/2406.04093) — TopK + AuxK recipe
- [SAELens](https://github.com/jbloomAus/SAELens) — our safetensors format

---

## License

MIT. Built by [Caio Vicentino](https://huggingface.co/caiovicentino1) + OpenInterpretability. 2026.

[openinterp.org](https://openinterp.org) · [github.com/OpenInterpretability](https://github.com/OpenInterpretability) · [hi@openinterp.org](mailto:hi@openinterp.org)
