# openinterp-cli

[![PyPI](https://img.shields.io/pypi/v/openinterp-cli.svg)](https://pypi.org/project/openinterp-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-openinterp.org-blue.svg)](https://openinterp.org/docs)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Python SDK + CLI for [openinterp.org](https://openinterp.org) — search the
feature **Atlas**, generate per-token **Traces**, and upload your own SAEs.

> **Status: v0.0.1 (Alpha).** The public surface is live and installable today.
> The full Atlas backend and on-device `generate_trace()` land in **v0.1.0 (Q2 2026)**.
> Until then the SDK ships a curated offline fallback so every example below runs.

---

## Install

```bash
pip install openinterp-cli
```

Requires Python 3.10+.

---

## Quickstart

### 1. Search the Atlas from the CLI

```bash
openinterp atlas "overconfidence"
```

```
                    Atlas results: 'overconfidence'
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ ID     ┃ Name                    ┃ Model             ┃ AUROC ┃ Description┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ f2503  │ overconfidence_pattern  │ Qwen/Qwen3.6-27B  │  0.54 │ Definitive…│
└────────┴─────────────────────────┴───────────────────┴───────┴────────────┘
```

### 2. Search from Python

```python
from openinterp import search_features

features = search_features("medical reasoning", limit=5)
for f in features:
    print(f"{f.id}  {f.name:30s}  AUROC={f.auroc}")
```

### 3. Generate a Trace (v0.1.0 preview)

```python
from openinterp import generate_trace

# Ships in v0.1.0 (Q2 2026). For now the call raises NotImplementedError with
# a pointer to the public interactive trace at openinterp.org/observatory/trace.
trace = generate_trace(
    model_id="Qwen/Qwen3.6-27B",
    prompt="The patient presents with acute chest pain.",
    sae_repo="caiovicentino1/qwen36-27b-sae-multilayer",
)
```

---

## CLI reference

```
openinterp --help
openinterp atlas <query> [--model HF_ID] [--limit N]
openinterp trace --model HF_ID --prompt TEXT [--sae-repo HF_ID]
openinterp --version
```

---

## Roadmap

| Version | Target   | Ships                                                          |
|---------|----------|----------------------------------------------------------------|
| 0.0.1   | Apr 2026 | Public API surface, Atlas search (curated fallback), CLI skeleton |
| 0.1.0   | Q2 2026  | Live Atlas backend, `generate_trace()`, `upload_trace()`       |
| 0.2.0   | Q3 2026  | SAE upload, cross-model feature diffing, notebook integration  |

---

## Links

- Homepage: <https://openinterp.org>
- Docs: <https://openinterp.org/docs>
- Observatory (live traces): <https://openinterp.org/observatory/trace>
- Repository: <https://github.com/OpenInterpretability/cli>
- Notebooks: <https://github.com/OpenInterpretability/notebooks>

---

## License

MIT — see [LICENSE](./LICENSE).

Built by [Caio Vicentino](https://openinterp.org) as part of the
OpenInterpretability project.
