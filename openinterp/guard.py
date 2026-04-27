"""FabricationGuard — activation-probe hallucination detection for open-weights LLMs.

This is the OpenInterp production probe-based guard. It wraps any HuggingFace
transformer with a hook at the SAE-supervised residual layer, captures the
last-token activation, and runs a small linear probe to score the prompt for
fabrication risk.

Three modes:

- ``detect``  — return the score, leave the model output untouched.
- ``warn``    — same as detect, plus a ``flagged`` bool when the score is high.
- ``abstain`` — replace high-score outputs with a calibrated uncertainty response.

Quick start::

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from openinterp import FabricationGuard

    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3.6-27B", torch_dtype="bfloat16",
        device_map="cuda", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

    guard = FabricationGuard.from_pretrained("Qwen/Qwen3.6-27B")
    guard.attach(model, tok)

    out = guard.generate("Who is Bambale Osby?", mode="abstain", threshold=0.7)
    print(out["text"])      # "I'm not confident about this..."
    print(out["score"])     # 0.93

Source paper / dataset:
https://huggingface.co/datasets/caiovicentino1/FabricationGuard-linearprobe-qwen36-27b

Requires ``pip install 'openinterp[full]'`` (torch + transformers + scikit-learn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


DEFAULT_PROBE_REGISTRY = {
    # model_name -> hf dataset repo holding probe.joblib + meta.json
    "Qwen/Qwen3.6-27B": "caiovicentino1/FabricationGuard-linearprobe-qwen36-27b",
    "qwen3.6-27b":       "caiovicentino1/FabricationGuard-linearprobe-qwen36-27b",
}


class FabricationGuardError(RuntimeError):
    """Raised on configuration or runtime failures of FabricationGuard."""


def _require_full() -> None:
    """Ensure the optional heavy dependencies are present."""
    missing: List[str] = []
    for mod in ("torch", "transformers", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        import joblib  # noqa: F401
    except ImportError:
        missing.append("joblib")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        raise FabricationGuardError(
            "FabricationGuard requires optional dependencies. Install with:\n"
            "    pip install 'openinterp[full]'\n"
            f"Missing: {', '.join(missing)}"
        )


@dataclass
class GuardOutput:
    """Result of ``FabricationGuard.generate(...)``."""

    text: str
    score: float
    flagged: bool
    mode: str
    abstained: bool
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": float(self.score),
            "flagged": bool(self.flagged),
            "mode": self.mode,
            "abstained": bool(self.abstained),
            "threshold": float(self.threshold),
        }


class FabricationGuard:
    """Activation-probe fabrication detector for an HF transformer.

    Construct via :meth:`from_pretrained` to download the matching probe
    artifact from HuggingFace, then call :meth:`attach` once the model
    is loaded. Use :meth:`score` for raw probabilities or :meth:`generate`
    to run model.generate with optional abstention.
    """

    DEFAULT_ABSTAIN_RESPONSE = (
        "I'm not confident about this. Please verify with an authoritative source."
    )

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        probe: Any,
        scaler: Any,
        layer: int,
        threshold: float,
        meta: Optional[Dict[str, Any]] = None,
        abstain_response: Optional[str] = None,
    ):
        self.probe = probe
        self.scaler = scaler
        self.layer = int(layer)
        self.threshold = float(threshold)
        self.meta = meta or {}
        self.abstain_response = abstain_response or self.DEFAULT_ABSTAIN_RESPONSE
        # Set on attach()
        self.model = None
        self.tok = None
        self.blocks = None
        self._buf: Dict[str, Any] = {}
        self._hook = None

    # -------------------------------------------------------- from_pretrained
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        probe_repo: Optional[str] = None,
        threshold: Optional[float] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "FabricationGuard":
        """Download a matching probe artifact for ``model_id`` and return a guard.

        Parameters
        ----------
        model_id
            HuggingFace model ID (e.g. ``"Qwen/Qwen3.6-27B"``). Used to look up
            the matching probe in the default registry.
        probe_repo
            Override the registry — point at any HF dataset that contains
            ``probe.joblib`` + ``meta.json`` in its root.
        threshold
            Override the calibrated threshold from the probe metadata.
        revision, token, cache_dir
            Forwarded to ``huggingface_hub.hf_hub_download``.
        """
        _require_full()
        import json
        import joblib
        from huggingface_hub import hf_hub_download

        repo = probe_repo or DEFAULT_PROBE_REGISTRY.get(model_id) \
            or DEFAULT_PROBE_REGISTRY.get(model_id.lower())
        if repo is None:
            raise FabricationGuardError(
                f"No probe registered for {model_id}. Pass probe_repo=... explicitly. "
                f"Known: {sorted(DEFAULT_PROBE_REGISTRY)}"
            )

        common_kwargs = dict(
            repo_id=repo, repo_type="dataset",
            revision=revision, token=token, cache_dir=cache_dir,
        )
        probe_path = hf_hub_download(filename="probe.joblib", **common_kwargs)
        meta_path  = hf_hub_download(filename="meta.json", **common_kwargs)

        artifacts = joblib.load(probe_path)
        meta = json.load(open(meta_path))
        layer = int(artifacts.get("layer") or meta.get("probe_layer") or
                    meta.get("layer", "31").lstrip("L"))
        if isinstance(meta.get("probe_layer"), str):
            layer = int(meta["probe_layer"].lstrip("L"))
        thr = threshold if threshold is not None else float(
            meta.get("best_threshold", 0.5)
        )
        return cls(
            probe=artifacts["probe"], scaler=artifacts["scaler"],
            layer=layer, threshold=thr, meta=meta,
        )

    # --------------------------------------------------------------- attach
    def attach(self, model: Any, tokenizer: Any) -> "FabricationGuard":
        """Register the residual-stream hook on ``model`` at ``self.layer``.

        Call this once the model has been loaded onto its device. Idempotent
        if the same model/tokenizer is passed again.
        """
        _require_full()
        self.model = model
        self.tok = tokenizer
        self.blocks = self._locate_blocks(model)
        if self._hook is not None:
            self._hook.remove()
        self._buf = {}
        self._hook = self.blocks[self.layer].register_forward_hook(self._capture_hook)
        return self

    @staticmethod
    def _locate_blocks(model: Any):
        candidates = [model]
        if hasattr(model, "base_model") and model.base_model is not model:
            candidates.append(
                model.base_model.model if hasattr(model.base_model, "model")
                else model.base_model
            )
        if hasattr(model, "model"):
            candidates.append(model.model)
        for start in candidates:
            for path in [
                ("model", "language_model", "layers"),
                ("language_model", "layers"),
                ("model", "layers"),
                ("layers",),
            ]:
                cur = start
                ok = True
                for p in path:
                    if hasattr(cur, p):
                        cur = getattr(cur, p)
                    else:
                        ok = False
                        break
                if ok and hasattr(cur, "__getitem__"):
                    return cur
        raise FabricationGuardError(
            "Could not locate transformer block list on this model. "
            "Pass blocks= kwarg manually if your model uses a non-standard layout."
        )

    def _capture_hook(self, _mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self._buf["h"] = h.detach()

    # ----------------------------------------------------------- close / dunder
    def close(self) -> None:
        """Remove the forward hook. Safe to call multiple times."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "FabricationGuard":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ score
    def score(
        self,
        prompt: Union[str, Sequence[str]],
        max_input_length: int = 512,
    ) -> Union[float, List[float]]:
        """Return the fabrication probability ∈ [0, 1] for one or more prompts.

        Higher score → higher likelihood that the model will fabricate when
        answering this prompt.
        """
        if self.model is None or self.tok is None:
            raise FabricationGuardError(
                "Guard not attached. Call .attach(model, tokenizer) first."
            )
        import torch

        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)

        enc = self.tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_length,
        ).to(self._device())

        with torch.no_grad():
            self._buf = {}
            self.model(**enc)
        h = self._buf.get("h")
        if h is None:
            raise FabricationGuardError("Hook did not capture residual — check attach().")

        last_pos = enc["attention_mask"].sum(dim=1) - 1
        last_h = h[torch.arange(h.size(0)), last_pos].float().cpu().numpy()
        h_scaled = self.scaler.transform(last_h)
        probs = self.probe.predict_proba(h_scaled)[:, 1]
        out = [float(p) for p in probs]
        return out[0] if single else out

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    # ----------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        mode: str = "detect",
        threshold: Optional[float] = None,
        max_new_tokens: int = 128,
        max_input_length: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        abstain_response: Optional[str] = None,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """Score the prompt, optionally short-circuit with an abstention, then generate.

        Parameters
        ----------
        prompt
            User-facing query.
        mode
            One of ``"detect"``, ``"warn"``, ``"abstain"``.
        threshold
            Score threshold for flagging. Defaults to the calibrated value
            from the probe metadata.
        abstain_response
            Override the canned uncertainty response.
        generate_kwargs
            Extra kwargs forwarded to ``model.generate``.
        """
        if mode not in {"detect", "warn", "abstain"}:
            raise FabricationGuardError(
                f"mode must be 'detect' | 'warn' | 'abstain' (got {mode!r})"
            )
        if self.model is None or self.tok is None:
            raise FabricationGuardError("Guard not attached. Call .attach(...) first.")

        thr = self.threshold if threshold is None else float(threshold)
        score = self.score(prompt, max_input_length=max_input_length)
        assert isinstance(score, float)
        flagged = score > thr

        if mode == "abstain" and flagged:
            return GuardOutput(
                text=abstain_response or self.abstain_response,
                score=score, flagged=True, mode=mode, abstained=True, threshold=thr,
            ).as_dict()

        # generate normally
        import torch
        enc = self.tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_input_length,
        ).to(self._device())
        with torch.no_grad():
            gen_ids = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                **generate_kwargs,
            )
        new_text = self.tok.decode(
            gen_ids[0, enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return GuardOutput(
            text=new_text, score=score,
            flagged=bool(flagged) and mode in {"warn", "abstain"},
            mode=mode, abstained=False, threshold=thr,
        ).as_dict()

    # ----------------------------------------------------------- repr
    def __repr__(self) -> str:
        attached = "attached" if self.model is not None else "detached"
        return (
            f"FabricationGuard(layer={self.layer}, threshold={self.threshold:.3f}, "
            f"{attached})"
        )
