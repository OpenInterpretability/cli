# Changelog

All notable changes to `openinterp` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
