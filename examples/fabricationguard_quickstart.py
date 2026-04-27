"""FabricationGuard quickstart — score and abstain in 30 lines.

Requires:  pip install 'openinterp[full]'
Hardware:  any GPU with >= 24 GB VRAM (Qwen3.6-27B in bf16 = ~54 GB; smaller GPU
           needs `device_map='auto'` with offloading or a smaller model).

Run:
    python examples/fabricationguard_quickstart.py
"""
import torch
from transformers import AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText as ModelCls
except ImportError:
    from transformers import AutoModelForCausalLM as ModelCls

from openinterp import FabricationGuard


MODEL_ID = "Qwen/Qwen3.6-27B"


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = ModelCls.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print("Loading FabricationGuard probe ...")
    guard = FabricationGuard.from_pretrained(MODEL_ID).attach(model, tok)
    print(f"  {guard}")

    prompts = [
        ("known",   "Who is Albert Einstein?"),
        ("obscure", "Who is Bambale Osby?"),
        ("synthetic", "Who is Vlasik Korpel?"),
    ]
    print("\nScoring + generating with abstain mode ...\n")
    for label, prompt in prompts:
        out = guard.generate(prompt, mode="abstain", max_new_tokens=60)
        print(f"[{label}]  score={out['score']:.3f}  "
              f"flagged={out['flagged']}  abstained={out['abstained']}")
        print(f"  Q: {prompt}")
        print(f"  A: {out['text'][:200]}\n")


if __name__ == "__main__":
    main()
