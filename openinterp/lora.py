"""LoRA save/load utilities for Qwen3.6 reasoning models.

Qwen3.6 PEFT-save creates state-dict keys with a `.language_model.` infix that
breaks downstream `PeftModel.from_pretrained()` calls — silently. The model
loads, no error is raised, but the adapter has zero effect (max logit-diff
between base and "loaded" model is exactly 0.000).

This module provides `safe_load_qwen36_lora()` that strips the infix and
verifies adapter application via a logit-diff sanity check before returning.

Usage:
    from openinterp.lora import safe_load_qwen36_lora

    model = safe_load_qwen36_lora(
        base_model_id="Qwen/Qwen3.6-27B",
        adapter_path="path/to/checkpoint-200",
        verify_prompt="The capital of France is",
    )

    # Without verification (faster, riskier):
    model = safe_load_qwen36_lora(
        base_model_id="Qwen/Qwen3.6-27B",
        adapter_path="path/to/checkpoint-200",
        verify=False,
    )

This bug was identified in nb39 → nb40 → nb41 v2 (April 2026) on Qwen3.6-27B
multi-probe DPO checkpoints. The fix was load-bearing for the paper-2 grokking
finding (otherwise no signal could be detected, since the adapter wasn't applied).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


class LoRAVerificationError(RuntimeError):
    """Raised when adapter load verification fails (logit-diff under tolerance)."""


def strip_language_model_infix(state_dict: dict) -> dict:
    """Strip `.language_model.` infix from PEFT-saved Qwen3.6 keys.

    Qwen3.6 PEFT-save produces keys like:
        base_model.model.language_model.layers.31.self_attn.q_proj.lora_A.weight

    `PeftModel.from_pretrained()` against a reloaded dense Qwen3.6 expects:
        base_model.model.layers.31.self_attn.q_proj.lora_A.weight

    Returns a new dict with the infix removed. Original dict is not mutated.
    """
    return {k.replace(".language_model.", "."): v for k, v in state_dict.items()}


def verify_adapter_loaded(
    base_model,
    loaded_model,
    tokenizer,
    prompt: str = "The capital of France is",
    tolerance: float = 0.01,
) -> float:
    """Verify adapter is actually applied via logit-diff.

    Returns the max absolute logit difference between base and loaded model
    on the given prompt. Raises LoRAVerificationError if diff < tolerance,
    which indicates the adapter loaded but produced no functional change
    (the silent-failure mode of the Qwen3.6 LoRA bug).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)
    with torch.no_grad():
        base_logits = base_model(**inputs).logits
        loaded_logits = loaded_model(**inputs).logits
    diff = (base_logits - loaded_logits).abs().max().item()
    if diff < tolerance:
        raise LoRAVerificationError(
            f"Adapter silently failed to load: max logit-diff={diff:.6f} "
            f"(below tolerance {tolerance}). "
            f"This is the Qwen3.6 .language_model. infix bug. "
            f"Strip the infix from your state dict before PeftModel.from_pretrained()."
        )
    return diff


def safe_load_qwen36_lora(
    base_model_id: str,
    adapter_path: Union[str, Path],
    verify: bool = True,
    verify_prompt: str = "The capital of France is",
    verify_tolerance: float = 0.01,
    base_model=None,
    tokenizer=None,
    torch_dtype=None,
    device_map: Optional[str] = "auto",
):
    """Load a Qwen3.6 LoRA adapter with the .language_model. fix + verification.

    Args:
        base_model_id: HuggingFace model ID, e.g. "Qwen/Qwen3.6-27B".
        adapter_path: Local path or HF repo for the saved adapter.
        verify: If True (default), run logit-diff sanity check after load.
        verify_prompt: Text used for verification logit comparison.
        verify_tolerance: Minimum acceptable max-logit-diff.
        base_model: Pre-loaded base model. If None, loads from base_model_id.
        tokenizer: Pre-loaded tokenizer. If None, loads from base_model_id.
        torch_dtype: dtype for base model load (default: bfloat16).
        device_map: device mapping (default: "auto").

    Returns:
        PeftModel with adapter applied, verified to differ from base.

    Raises:
        LoRAVerificationError: If verify=True and adapter produced no logit-diff.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    adapter_path = Path(adapter_path)

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch_dtype, device_map=device_map
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Detect & fix the .language_model. infix in saved adapter weights
    adapter_file = adapter_path / "adapter_model.safetensors"
    if adapter_file.exists():
        state = load_file(str(adapter_file))
        if any(".language_model." in k for k in state.keys()):
            fixed = strip_language_model_infix(state)
            # Save fixed state to a sibling file for PeftModel to load
            fixed_path = adapter_path / "adapter_model.safetensors"
            save_file(fixed, str(fixed_path))

    # Load adapter onto base model
    loaded_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # Verify
    if verify:
        diff = verify_adapter_loaded(
            base_model, loaded_model, tokenizer,
            prompt=verify_prompt, tolerance=verify_tolerance,
        )
        # Ok — adapter is functional
    return loaded_model


__all__ = [
    "strip_language_model_infix",
    "verify_adapter_loaded",
    "safe_load_qwen36_lora",
    "LoRAVerificationError",
]
