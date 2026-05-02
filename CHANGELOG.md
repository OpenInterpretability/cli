# Changelog

All notable changes to `openinterp` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.2] — 2026-05-01

### Fixed — `safe_load_qwen36_lora` false-positive verification

The `verify=True` path of `safe_load_qwen36_lora()` was raising
`LoRAVerificationError` even when the adapter loaded correctly.

Root cause: `PeftModel.from_pretrained(base_model, ...)` mutates `base_model`
in-place by injecting LoRA layers. The previous code captured `base_logits`
AFTER this mutation, so the "base" reference was already the LoRA-applied
model. Comparing it against `loaded_logits` from the same object produced
`diff = 0.000`, which the verifier flagged as silent failure — but in
reality the adapter was working.

Fix: capture `base_logits` BEFORE calling `PeftModel.from_pretrained()`.
This produces an honest reference for the diff comparison.

Discovered during nb44 v2 (paper-3 behavior eval) on 2026-05-01.

## [0.2.1] — 2026-05-01

### Added — `openinterp.lora` module

- **`safe_load_qwen36_lora(base_model_id, adapter_path, ...)`** — safe loader
  for Qwen3.6 LoRA adapters that auto-strips the `.language_model.` infix from
  PEFT-saved state-dict keys and verifies via logit-diff sanity check.
  Discovered in nb39 → nb40 → nb41 v2 (April 2026): without the strip,
  `PeftModel.from_pretrained()` against a reloaded dense Qwen3.6 silently
  fails — adapter loaded, max logit-diff = `0.000`, no error raised.
- **`strip_language_model_infix(state_dict)`** — pure dict transform exposed
  for users who handle their own load pipeline.
- **`verify_adapter_loaded(base, loaded, tokenizer, ...)`** — standalone
  sanity check returning max logit-diff between base and loaded models.
- **`LoRAVerificationError`** — raised when adapter loaded but produced no
  functional change (the silent-failure mode of the bug).

```python
from openinterp import safe_load_qwen36_lora

model = safe_load_qwen36_lora(
    base_model_id="Qwen/Qwen3.6-27B",
    adapter_path="path/to/checkpoint-200",
)  # auto strip + auto verify
```

This bug invalidated about 10 hours of prior eval work on our paper-2 (probe-detected
grokking in multi-probe DPO) before being caught. Anyone working with Qwen3.6 LoRA
save/reload pipelines should run the sanity check.

## [0.2.0] — 2026-04-27

### Added — FabricationGuard

- **`FabricationGuard` class** for activation-probe hallucination detection on
  open-weights LLMs. AUROC 0.88 cross-task on SimpleQA, **−88% confident-wrong
  reduction** on factual QA, **~1 ms** scoring latency. Apache-2.0.
  - `FabricationGuard.from_pretrained(model_id)` downloads the matching probe
    from HuggingFace (initial registry entry: `Qwen/Qwen3.6-27B` →
    [`caiovicentino1/FabricationGuard-linearprobe-qwen36-27b`](https://huggingface.co/datasets/caiovicentino1/FabricationGuard-linearprobe-qwen36-27b)).
  - `.attach(model, tokenizer)` registers a forward hook at the probe layer
    (L31 for Qwen3.6-27B). Idempotent; safe to re-attach. Hook lifecycle is
    managed via `.close()` / context manager / `__del__`.
  - `.score(prompt)` returns fabrication probability ∈ [0, 1] in ~1 ms.
  - `.generate(prompt, mode=…)` runs `model.generate` with optional abstention.
    Modes: `detect` | `warn` | `abstain`.
  - `GuardOutput` dataclass with structured response: `text`, `score`,
    `flagged`, `mode`, `abstained`, `threshold`.
  - Custom `abstain_response` override + arbitrary `generate_kwargs` pass-through.
  - Auto layer-list discovery handles dense + multimodal + hybrid GDN layouts.
  - Helpful `FabricationGuardError` raised when `[full]` extras missing.
- **`openinterp guard ...`** CLI command. Loads model, attaches probe, scores,
  optionally generates with abstention. `--json` for machine output.
- `scikit-learn>=1.3` and `joblib>=1.3` added to the `[full]` extra (both
  needed to load the probe artifact).
- `tests/test_guard.py` — structural tests + invariants.

### Changed
- `__init__.py` now exports `FabricationGuard`, `FabricationGuardError`,
  `GuardOutput`.
- Package keywords expanded: `hallucination`, `fabrication`, `guard`,
  `linear probe`.

### Source artifacts
- Probe + headline figure: <https://huggingface.co/datasets/caiovicentino1/FabricationGuard-linearprobe-qwen36-27b>
- Reproducer notebooks (open-source): `OpenInterpretability/notebooks/30_hallucinationguard_proof_qwen36_27b.ipynb`
  and `31_hallucinationguard_v2_linear_probe.ipynb`.

### Planned for 0.3.0
- Multi-model probes via Pearson_CE cross-model transfer (Llama-3.3, Gemma-2,
  Mistral). Methodology in [`gemma2-2b-crosscoder-model-diff-papergrade`](https://huggingface.co/caiovicentino1/gemma2-2b-crosscoder-model-diff-papergrade).
- `openinterp score`, `openinterp steer`, `openinterp circuit`,
  `openinterp publish` wrapping notebooks 18/06/14/15.
- vLLM + SGLang inference plugins.
- LangChain + LlamaIndex middleware.

---

## [0.1.0] — 2026-04-23

### Added
- Package renamed from `openinterp-cli` to **`openinterp`** on PyPI.
- **`generate_trace(...)` — real implementation.** Loads any HF model +
  SAE (sae_lens-format safetensors on HuggingFace), runs a forward
  pass, captures residual-stream activations at a target layer,
  applies the SAE, and emits a `Trace` object matching the
  openinterp.org Trace Theater schema byte-for-byte.
- **`openinterp trace ...`** CLI command.
- **`openinterp info`** shows installed version + optional-dep status.
- Optional `[full]` install extra that pulls `torch`, `transformers`,
  `safetensors`, `accelerate`, `numpy` for trace generation.
- Fallback layer-discovery that supports Llama/Gemma/Qwen/Mistral/Phi
  (`model.model.layers`) + nested multimodal paths
  (`model.model.language_model.layers`) + GPT-2 (`model.transformer.h`).
- Optional `--catalog feature_catalog.json` flag to attach per-feature
  names and descriptions from notebook 04.
- Upgraded to Pydantic v2 style, added `TraceFeature` model.

### Changed
- `TraceUnavailable` exception replaces `NotImplementedError` on
  missing deps — with a helpful pointer to `pip install "openinterp[full]"`.

### Planned for 0.2.0 (Q2 2026)
- `upload_trace()` → shareable openinterp.org URL.
- `openinterp score`, `openinterp steer`, `openinterp circuit`,
  `openinterp publish` wrapping notebooks 18/06/14/15.

## [0.0.1] — 2026-04-23 (as `openinterp-cli`)

Initial alpha. `search_features` with offline fallback. `generate_trace`
raised `NotImplementedError`.
