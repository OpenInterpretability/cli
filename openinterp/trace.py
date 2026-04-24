"""Trace generation — real transformers-based implementation.

Loads a model + an SAE from HuggingFace, runs a forward pass on a prompt,
captures residual-stream activations at a target layer, applies the SAE, and
emits a `Trace` object matching the openinterp.org Trace Theater schema.

Requires `pip install openinterp[full]` (torch + transformers + safetensors).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from openinterp.models import Trace, TraceFeature


class TraceUnavailable(RuntimeError):
    """Raised when the optional [full] dependencies are missing."""


def _require_full() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        raise TraceUnavailable(
            "generate_trace requires optional dependencies. "
            "Install with:  pip install 'openinterp[full]'"
        ) from e


def _load_sae(sae_repo: str, layer: int, d_in: int, d_sae: int, k: int):
    """Load a TopK SAE from HuggingFace."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    class TopKSAE(nn.Module):
        def __init__(self, d_in: int, n: int, k: int):
            super().__init__()
            self.W_enc = nn.Parameter(torch.zeros(d_in, n))
            self.W_dec = nn.Parameter(torch.zeros(n, d_in))
            self.b_enc = nn.Parameter(torch.zeros(n))
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.k = k

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            top_v, top_i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, top_i, F.relu(top_v))
            return z

    candidates = [
        f"sae_L{layer}_latest.safetensors",
        f"sae_L{layer}.safetensors",
        "sae_final.safetensors",
        "sae.safetensors",
    ]
    weights = None
    last_err: Optional[Exception] = None
    for fname in candidates:
        try:
            path = hf_hub_download(sae_repo, fname)
            weights = load_file(path)
            break
        except Exception as e:
            last_err = e
    if weights is None:
        raise FileNotFoundError(
            f"Could not find SAE weights in {sae_repo}. "
            f"Tried: {candidates}. Last error: {last_err}"
        )

    sae = TopKSAE(d_in=d_in, n=d_sae, k=k)
    sae.load_state_dict(weights, strict=False)
    sae.eval()
    return sae


def _get_layer(model, layer_idx: int):
    """Discover the decoder layer at layer_idx — handles nested architectures."""
    for path in [
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
    ]:
        try:
            cur = model
            for p in path:
                cur = getattr(cur, p)
            return cur[layer_idx]
        except AttributeError:
            continue
    raise RuntimeError("Could not locate decoder layers")


def generate_trace(
    model_id: str,
    prompt: str,
    sae_repo: str,
    layer: int,
    d_model: int = 2304,
    d_sae: int = 16384,
    k: int = 32,
    max_new_tokens: int = 30,
    top_n_features: int = 10,
    device: Optional[str] = None,
    feature_catalog: Optional[dict] = None,
) -> Trace:
    """Generate a Trace from model + prompt + SAE.

    Args:
        model_id: HF model ID, e.g. "Qwen/Qwen3.6-27B" or "google/gemma-2-2b".
        prompt: the input prompt.
        sae_repo: HF repo containing the SAE weights (sae_L{layer}_latest.safetensors).
        layer: target residual-stream layer index.
        d_model: base model hidden dim (default 2304 for Gemma-2-2B).
        d_sae: SAE dictionary size.
        k: TopK sparsity.
        max_new_tokens: tokens to generate.
        top_n_features: number of features to include in the trace (ranked by total activation).
        device: "cuda" / "cpu" / None (auto-select).
        feature_catalog: optional {"features": [{"id": ..., "name": ..., "desc": ..., "auroc": ...}]} for labels.

    Returns:
        Trace — matches openinterp.org Trace Theater schema, ready to JSON-serialize.

    Raises:
        TraceUnavailable: if `pip install openinterp[full]` wasn't run.
    """
    _require_full()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer + model
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load SAE
    sae = _load_sae(sae_repo, layer, d_model, d_sae, k).to(device, torch.float32)

    # Install hook
    target = _get_layer(model, layer)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[:, -1, :].detach().float())  # last-token residual per forward

    handle = target.register_forward_hook(hook)
    try:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        new_ids = out[0, ids.shape[1]:]
        tokens = [tok.decode([int(t)], skip_special_tokens=False) for t in new_ids]
    finally:
        handle.remove()

    # captured contains (prompt_forward + 1-per-generated-token) residuals at last position
    # Keep the per-generated-token residuals (last max_new_tokens entries)
    residuals = torch.stack(captured[-len(tokens):])  # (T, d_model)

    # SAE encode
    with torch.no_grad():
        z = sae.encode(residuals)  # (T, d_sae)

    # Pick top features by total activation
    total_act = z.sum(dim=0)
    top_idx = total_act.topk(top_n_features).indices.tolist()

    # Build per-feature per-token activations, normalized to [0, 1]
    per_feature_act = z[:, top_idx].T  # (top_n, T)
    max_per_row = per_feature_act.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    normed = (per_feature_act / max_per_row).clamp(0, 1).tolist()

    # Assemble feature entries (with optional catalog lookup)
    catalog = {f["id"]: f for f in (feature_catalog or {}).get("features", [])}
    features = []
    for fi in top_idx:
        fid = f"f{fi}"
        entry = catalog.get(fid, {})
        features.append(TraceFeature(
            id=fid,
            name=entry.get("name", f"feature_{fi}"),
            desc=entry.get("desc", f"Feature {fi} — label via 04_discover_features.ipynb"),
            auroc=entry.get("auroc", 0.0),
        ))

    # Round activations to 4 decimals
    activations = [[round(v, 4) for v in row] for row in normed]

    trace = Trace(
        prompt=prompt,
        model=model_id,
        layer=f"L{layer} residual",
        sae_repo=sae_repo,
        tokens=tokens,
        features=features,
        activations=activations,
        counterfactuals={},
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Clean up VRAM for CLI users who run back-to-back
    try:
        del model, sae, residuals, z, per_feature_act
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return trace


def upload_trace(trace: Trace, public: bool = True) -> str:
    """Upload a Trace to openinterp.org; returns a shareable URL.

    Coming in v0.2.0 (Q2 2026) — requires the upload endpoint to be live at
    openinterp.org/api/trace/upload.
    """
    raise NotImplementedError(
        "upload_trace() ships in v0.2.0 (Q2 2026). "
        "For now: save your trace to a HuggingFace SAE repo and share the raw URL — "
        "see https://openinterp.org/observatory/trace"
    )
