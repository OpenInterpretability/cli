# AgentProbeGuard env-coupling cookbook

This guide is for deployments where probe weights were trained in one environment
and run in another (different attention implementation, kernel stack, or
transformers runtime).

Use this page to:

1. detect env coupling before rollout;
2. refit with `AgentProbeGuard.refit()` when transfer degrades;
3. decide between one refit helper and per-env weight variants.

## What "env coupling" means

Probe weights are fit on a specific residual distribution. That distribution can
shift when inference changes from one attention path to another (`fla`, `sdpa`,
`flash`) or when runtime versions differ.

Typical symptom: thresholds that were calibrated in training skew toward
`proceed` in production, even when the signal is still present.

## 1) Detect coupling in the target environment

Run a held-out labeled set in the exact target stack and compare to training:

- AUROC drop from source to target;
- cosine drift of residual directions used by the probes;
- residual-norm shift at probe layers.

If AUROC and direction metrics are stable, reuse is often fine. If they drift,
refit in-target before routing production traffic.

## 2) Refit with `AgentProbeGuard.refit()`

Minimal example:

```python
from openinterp import AgentProbeGuard

agent_probe_guard = AgentProbeGuard.from_pretrained("Qwen/Qwen3.6-27B")
agent_probe_guard.attach(model, tok)

metrics = agent_probe_guard.refit(
    prompts=prompts,   # list[str] or chat messages
    labels=labels,     # aligned 0/1 labels
    verbose=True,
)

print(metrics)
```

Use labels that match your routing objective in the target environment.
After refit, re-check AUROC and threshold behavior before deployment.

## 3) One refit helper vs per-env variants

Prefer one refit helper when:

- all inference traffic uses one stack;
- AUROC/cosine stay stable across canary batches;
- operational simplicity matters more than env-specific tuning.

Prefer per-env weight variants when:

- you deploy on multiple stacks (`fla`/`sdpa`/`flash`) with measurable drift;
- thresholds diverge across environments;
- compliance or reliability policy requires per-env validation artifacts.

## Caveat

Even with the same model and prompt, residual directions can drift across
attention implementations. Treat transfer as empirical, not guaranteed.
Validate AUROC and cosine metrics in each target environment before rollout.
