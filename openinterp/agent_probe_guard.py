"""AgentProbeGuard — mid-reasoning detection for code agents on open-weights LLMs.

Two-probe activation gate for LLM-based code agents. Hooks the residual stream
of an HF transformer at two positions and reads light linear probes to predict
(a) whether the agent will succeed at its current SWE-bench-style task and
(b) whether the prompt would have triggered chain-of-thought thinking under a
permissive template.

Three decision modes:

- ``proceed``  — high probe score, continue the agent normally
- ``escalate`` — moderate score, route to a stronger model / human review
- ``skip``     — low score, abort the trace before burning the budget

This module is deliberately **detect-only**. We confirmed across three
intervention experiments (Phase 7 single-shot + continuous + Phase 8 amplitude
diagnostic up to α=+200) that the underlying probe directions are
epiphenomenal: they correlate with the outcome without participating in the
causal pathway that produces it. AgentProbeGuard exposes the read; it does not
attempt to bias the model.

Quick start::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openinterp import AgentProbeGuard

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype="bfloat16",
        device_map="cuda", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

    guard = AgentProbeGuard.from_pretrained("Qwen/Qwen3.6-27B")
    guard.attach(model, tok)

    decision = guard.assess(messages, partial_response=current_thought)
    if decision.action == "skip":
        raise BudgetSkip(decision.reason)
    elif decision.action == "escalate":
        return stronger_model.complete(messages)
    # else proceed normally

Source eval / dataset:
https://huggingface.co/datasets/caiovicentino1/agent-probe-guard-qwen36-27b

Requires ``pip install 'openinterp[full]'`` (torch + transformers + scikit-learn).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


DEFAULT_PROBE_REGISTRY = {
    "Qwen/Qwen3.6-27B": "caiovicentino1/agent-probe-guard-qwen36-27b",
    "qwen3.6-27b":      "caiovicentino1/agent-probe-guard-qwen36-27b",
}


class AgentProbeGuardError(RuntimeError):
    """Raised on configuration or runtime failures of AgentProbeGuard."""


def _require_full() -> None:
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
        raise AgentProbeGuardError(
            "AgentProbeGuard requires optional dependencies. Install with:\n"
            "    pip install 'openinterp[full]'\n"
            f"Missing: {', '.join(missing)}"
        )


@dataclass
class Decision:
    """Result of :meth:`AgentProbeGuard.assess`.

    The ``action`` is the recommended routing — ``proceed``, ``escalate``, or
    ``skip`` — derived from the underlying probe scores and configured
    thresholds. ``scores`` exposes the raw per-probe values so callers can
    implement custom routing if the defaults don't fit their cost model.
    """
    action: str  # "proceed" | "escalate" | "skip"
    reason: str
    scores: Dict[str, float]
    thresholds: Dict[str, float]
    fired: List[str]  # which probes contributed to this decision

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "scores": dict(self.scores),
            "thresholds": dict(self.thresholds),
            "fired": list(self.fired),
        }


class AgentProbeGuard:
    """Two-probe mid-reasoning gate for LLM-based code agents.

    The guard owns two forward hooks (one per probe layer) on the wrapped
    model. Both hooks capture the last-position residual on each forward pass
    and store it on the guard instance; the probe is then run on demand in
    :meth:`score_capability` (L43 pre-tool) or :meth:`score_thinking`
    (L55 last-prompt-token).

    High-level routing is provided by :meth:`assess`, which combines both
    scores under the configured thresholds and returns a :class:`Decision`.
    """

    DEFAULT_THRESHOLDS = {
        "skip_below": 0.20,       # below this on capability probe → skip
        "escalate_below": 0.50,   # between skip_below and this → escalate
        "thinking_low": 0.30,     # below this on thinking probe → suppressed-intent flag
    }

    def __init__(
        self,
        capability_probe: Any,
        capability_scaler: Any,
        capability_layer: int,
        capability_dims: Sequence[int],
        thinking_probe: Any,
        thinking_scaler: Any,
        thinking_layer: int,
        thinking_dims: Sequence[int],
        thresholds: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.capability_probe = capability_probe
        self.capability_scaler = capability_scaler
        self.capability_layer = int(capability_layer)
        self.capability_dims = tuple(int(d) for d in capability_dims)

        self.thinking_probe = thinking_probe
        self.thinking_scaler = thinking_scaler
        self.thinking_layer = int(thinking_layer)
        self.thinking_dims = tuple(int(d) for d in thinking_dims)

        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.meta = meta or {}

        self.model = None
        self.tok = None
        self.blocks = None
        self._buf: Dict[int, Any] = {}
        self._hooks: List[Any] = []

    # ----------------------------------------------------------- from_pretrained
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        probe_repo: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "AgentProbeGuard":
        """Download the matching probe artifact for ``model_id`` and return a guard.

        The HF dataset is expected to contain three files at its root:

        - ``probe_L43_pre_tool.joblib``  — sklearn pipeline + selected dim indices
        - ``probe_L55_thinking.joblib``  — sklearn pipeline + selected dim indices
        - ``meta.json``                  — layer indices, dim arrays, thresholds, eval metrics
        """
        _require_full()
        import json
        import joblib
        from huggingface_hub import hf_hub_download

        repo = probe_repo or DEFAULT_PROBE_REGISTRY.get(model_id) \
            or DEFAULT_PROBE_REGISTRY.get(model_id.lower())
        if repo is None:
            raise AgentProbeGuardError(
                f"No probe registered for {model_id}. Pass probe_repo=... explicitly. "
                f"Known: {sorted(DEFAULT_PROBE_REGISTRY)}"
            )

        common = dict(
            repo_id=repo, repo_type="dataset",
            revision=revision, token=token, cache_dir=cache_dir,
        )
        cap_path  = hf_hub_download(filename="probe_L43_pre_tool.joblib", **common)
        thk_path  = hf_hub_download(filename="probe_L55_thinking.joblib", **common)
        meta_path = hf_hub_download(filename="meta.json", **common)

        cap = joblib.load(cap_path)
        thk = joblib.load(thk_path)
        meta = json.load(open(meta_path))

        thresh = thresholds or meta.get("thresholds") or cls.DEFAULT_THRESHOLDS

        return cls(
            capability_probe=cap["probe"],
            capability_scaler=cap["scaler"],
            capability_layer=int(cap.get("layer") or meta["capability"]["layer"]),
            capability_dims=cap.get("dims") or meta["capability"]["dims"],
            thinking_probe=thk["probe"],
            thinking_scaler=thk["scaler"],
            thinking_layer=int(thk.get("layer") or meta["thinking"]["layer"]),
            thinking_dims=thk.get("dims") or meta["thinking"]["dims"],
            thresholds=thresh,
            meta=meta,
        )

    # --------------------------------------------------------------- attach
    def attach(self, model: Any, tokenizer: Any) -> "AgentProbeGuard":
        """Register forward hooks on both probe layers.

        Idempotent: re-calling on the same model removes prior hooks first.
        """
        _require_full()
        self.detach()
        self.model = model
        self.tok = tokenizer
        self.blocks = self._locate_blocks(model)
        self._buf = {}
        for layer_idx in (self.capability_layer, self.thinking_layer):
            h = self.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)
        return self

    def detach(self) -> None:
        """Remove forward hooks. Safe to call multiple times."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def __enter__(self) -> "AgentProbeGuard":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()

    def __del__(self):
        try:
            self.detach()
        except Exception:
            pass

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
        raise AgentProbeGuardError(
            "Could not locate transformer block list on this model. "
            "Pass blocks= kwarg manually if your model uses a non-standard layout."
        )

    def _make_hook(self, layer_idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._buf[layer_idx] = h.detach()
        return hook

    # ----------------------------------------------------------- internals
    def _device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _forward_and_capture(self, prompt_text: str, max_input_length: int = 4096):
        if self.model is None or self.tok is None:
            raise AgentProbeGuardError("Guard not attached. Call .attach(model, tok) first.")
        import torch
        enc = self.tok(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=max_input_length,
            add_special_tokens=False,
        ).to(self._device())
        with torch.no_grad():
            self._buf = {}
            self.model(**enc, use_cache=False)
        return enc

    def _score_at(
        self,
        prompt_text: str,
        layer: int,
        dims: Tuple[int, ...],
        scaler: Any,
        probe: Any,
        max_input_length: int = 4096,
    ) -> float:
        import numpy as np
        enc = self._forward_and_capture(prompt_text, max_input_length=max_input_length)
        h = self._buf.get(layer)
        if h is None:
            raise AgentProbeGuardError(
                f"Forward hook on layer {layer} did not capture residual. "
                "Did .attach() succeed?"
            )
        # Last attended position (handles padding)
        attn = enc.get("attention_mask")
        if attn is None:
            last_pos = h.shape[1] - 1
            last_h = h[0, last_pos].float().cpu().numpy().reshape(1, -1)
        else:
            last_idx = int(attn[0].sum().item()) - 1
            last_h = h[0, last_idx].float().cpu().numpy().reshape(1, -1)
        last_h = last_h[:, list(dims)]
        last_h = scaler.transform(last_h)
        prob = probe.predict_proba(last_h)[0, 1]
        return float(prob)

    # ----------------------------------------------------------- score_*
    def score_capability(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        partial_response: str = "",
        prompt_text: Optional[str] = None,
        max_input_length: int = 4096,
    ) -> float:
        """Probe the L43 pre-tool position for trace-success likelihood.

        Provide either ``messages`` (chat-style list of role/content dicts) plus
        the agent's ``partial_response`` so far, or a fully-rendered
        ``prompt_text``. Returns the probability ∈ [0, 1] that this trace will
        produce a successful patch.

        Higher score → more likely to succeed. Use the returned value with
        :meth:`assess` thresholds (skip_below=0.20, escalate_below=0.50) or
        roll your own.
        """
        text = prompt_text if prompt_text is not None else self._render_chat(messages or [], partial_response)
        return self._score_at(
            text, self.capability_layer, self.capability_dims,
            self.capability_scaler, self.capability_probe, max_input_length,
        )

    def score_thinking(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt_text: Optional[str] = None,
        max_input_length: int = 4096,
    ) -> float:
        """Probe the L55 last-prompt-token for suppressed thinking intent.

        Returns the probability ∈ [0, 1] that under a permissive template the
        model would have continued chain-of-thought reasoning. Useful for
        deciding whether a no-think configuration is hurting answer quality on
        a particular query.

        Note: this is a counterfactual signal — it reads what the model would
        have done, not a controllable lever.
        """
        text = prompt_text if prompt_text is not None else self._render_chat(messages or [], "")
        return self._score_at(
            text, self.thinking_layer, self.thinking_dims,
            self.thinking_scaler, self.thinking_probe, max_input_length,
        )

    # ----------------------------------------------------------- assess
    def assess(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        partial_response: str = "",
        prompt_text: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        max_input_length: int = 4096,
    ) -> Decision:
        """Run both probes and return a routing decision.

        Decision logic (default thresholds):

        - capability < ``skip_below`` (0.20): ``skip`` (high failure risk)
        - capability < ``escalate_below`` (0.50): ``escalate``
        - capability ≥ 0.50: ``proceed``

        The thinking probe is reported alongside but does not override the
        capability decision. Callers that want to use it as a gate (e.g. fall
        back to a thinking-enabled endpoint when ``thinking_low`` triggers)
        can read it from ``decision.scores['thinking']`` directly.
        """
        thr = dict(self.thresholds)
        if thresholds:
            thr.update(thresholds)

        cap_score = self.score_capability(
            messages=messages, partial_response=partial_response,
            prompt_text=prompt_text, max_input_length=max_input_length,
        )
        # Thinking probe uses the prompt without partial_response
        thk_score = self.score_thinking(
            messages=messages,
            prompt_text=prompt_text if partial_response == "" else None,
            max_input_length=max_input_length,
        )

        if cap_score < thr["skip_below"]:
            action = "skip"
            reason = (
                f"capability score {cap_score:.3f} < skip_below {thr['skip_below']:.2f} "
                "— trace unlikely to produce a patch"
            )
        elif cap_score < thr["escalate_below"]:
            action = "escalate"
            reason = (
                f"capability score {cap_score:.3f} ∈ [{thr['skip_below']:.2f}, "
                f"{thr['escalate_below']:.2f}) — recommend stronger model"
            )
        else:
            action = "proceed"
            reason = f"capability score {cap_score:.3f} ≥ escalate_below {thr['escalate_below']:.2f}"

        return Decision(
            action=action,
            reason=reason,
            scores={"capability": cap_score, "thinking": thk_score},
            thresholds=thr,
            fired=["L{}_pre_tool".format(self.capability_layer),
                   "L{}_thinking".format(self.thinking_layer)],
        )

    # ----------------------------------------------------------- helpers
    def _render_chat(self, messages: List[Dict[str, str]], partial_response: str) -> str:
        if not self.tok:
            raise AgentProbeGuardError("Guard not attached. Call .attach(model, tok) first.")
        if not messages:
            raise AgentProbeGuardError("Pass messages=... or prompt_text=...")
        try:
            base = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as exc:
            raise AgentProbeGuardError(
                f"Tokenizer chat template failed: {exc}. "
                "Pass prompt_text= explicitly to skip the template."
            )
        return base + (partial_response or "")

    # ----------------------------------------------------------- refit
    def refit(
        self,
        prompts: Sequence[Union[str, List[Dict[str, str]]]],
        labels: Sequence[int],
        capability_K: Optional[int] = None,
        thinking_K: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Refit both probes on the current inference environment.

        Probe weights from :meth:`from_pretrained` are coupled to the residual
        distribution of the environment they were captured in (attention
        implementation, fla/flash kernels, transformers version). When that
        environment differs from yours, expect AUROC to drop by 5-15 points
        and threshold-based routing to skew toward "proceed" — even though the
        underlying signal remains. This method captures fresh activations on
        your model and re-fits the top-K probes in place, replacing the loaded
        weights with environment-matched ones.

        Parameters
        ----------
        prompts
            List of prompts. Each can be a fully-rendered string OR a chat
            messages list (will be rendered via ``apply_chat_template``).
        labels
            Binary labels (0/1) aligned with ``prompts``. For the capability
            probe these should be patch-success / trace-success indicators;
            for the thinking probe, ``has_think_v1`` (auto-injected
            continuation past ``<think>``).
        capability_K, thinking_K
            Override the K values from the loaded probes. Defaults preserve
            the original capacities (10, 5).

        Returns
        -------
        dict with ``capability_auroc`` and ``thinking_auroc`` from 4-fold
        cross-validation on the refit data. AUROCs ≥ 0.80 indicate signal is
        real on this env; if both fall below 0.70, capture quality may need
        investigation.
        """
        _require_full()
        if self.model is None or self.tok is None:
            raise AgentProbeGuardError("Guard not attached. Call .attach(model, tok) first.")

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        if len(prompts) != len(labels):
            raise AgentProbeGuardError(
                f"len(prompts)={len(prompts)} != len(labels)={len(labels)}"
            )
        y = np.asarray([int(yi) for yi in labels])

        cap_K = int(capability_K) if capability_K is not None else len(self.capability_dims)
        thk_K = int(thinking_K) if thinking_K is not None else len(self.thinking_dims)

        if verbose:
            print(f"Refit on N={len(y)} prompts (capability K={cap_K}, thinking K={thk_K})")

        caps_cap, caps_thk = [], []
        for i, p in enumerate(prompts):
            if isinstance(p, str):
                text = p
            else:
                text = self._render_chat(p, "")
            self._forward_and_capture(text)
            for buf, layer, lst in (
                (self._buf, self.capability_layer, caps_cap),
                (self._buf, self.thinking_layer, caps_thk),
            ):
                h = buf[layer]
                # Use last attended position
                last_idx = h.shape[1] - 1
                lst.append(h[0, last_idx].float().cpu().numpy())
            if verbose and (i + 1) % 50 == 0:
                print(f"  captured {i+1}/{len(prompts)}")

        X_cap = np.stack(caps_cap)
        X_thk = np.stack(caps_thk)

        def _topk_diff(X, y, k):
            d = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))
            return np.argsort(-d)[:k]

        def _fit(X, y, K, label):
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            cv = []
            for tr, te in skf.split(X, y):
                d_tr = _topk_diff(X[tr], y[tr], K)
                sc = StandardScaler()
                Xtr = np.nan_to_num(sc.fit_transform(X[tr][:, d_tr]), nan=0.0, posinf=0.0, neginf=0.0)
                Xte = np.nan_to_num(sc.transform(X[te][:, d_tr]), nan=0.0, posinf=0.0, neginf=0.0)
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                clf.fit(Xtr, y[tr])
                cv.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
            auroc = float(np.mean(cv))

            # Final fit on all data using fold-stable selection
            dims = _topk_diff(X, y, K)
            sc = StandardScaler()
            Xs = np.nan_to_num(sc.fit_transform(X[:, dims]), nan=0.0, posinf=0.0, neginf=0.0)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(Xs, y)
            return clf, sc, tuple(int(d) for d in dims), auroc

        cap_clf, cap_sc, cap_dims, cap_auroc = _fit(X_cap, y, cap_K, "capability")
        thk_clf, thk_sc, thk_dims, thk_auroc = _fit(X_thk, y, thk_K, "thinking")

        # Replace loaded probes in place
        self.capability_probe = cap_clf
        self.capability_scaler = cap_sc
        self.capability_dims = cap_dims
        self.thinking_probe = thk_clf
        self.thinking_scaler = thk_sc
        self.thinking_dims = thk_dims
        self.meta = dict(self.meta)
        self.meta["refit"] = {
            "n": int(len(y)),
            "capability_auroc": cap_auroc,
            "thinking_auroc": thk_auroc,
            "capability_dims": list(cap_dims),
            "thinking_dims": list(thk_dims),
        }

        if verbose:
            print(f"capability AUROC (refit, 4-fold CV): {cap_auroc:.4f}")
            print(f"thinking AUROC (refit, 4-fold CV): {thk_auroc:.4f}")
            print(f"new capability dims: {list(cap_dims)}")
            print(f"new thinking dims: {list(thk_dims)}")

        return {"capability_auroc": cap_auroc, "thinking_auroc": thk_auroc}

    # ----------------------------------------------------------- repr
    def __repr__(self) -> str:
        attached = "attached" if self.model is not None else "detached"
        return (
            "AgentProbeGuard(capability=L{cap}+K{capk}, thinking=L{thk}+K{thkk}, "
            "thresholds=skip<{sb:.2f}/escalate<{eb:.2f}, {state})"
        ).format(
            cap=self.capability_layer, capk=len(self.capability_dims),
            thk=self.thinking_layer, thkk=len(self.thinking_dims),
            sb=self.thresholds["skip_below"], eb=self.thresholds["escalate_below"],
            state=attached,
        )
