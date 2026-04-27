"""openinterp — command line interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from openinterp import __version__
from openinterp.atlas import search_features
from openinterp.trace import generate_trace, TraceUnavailable
from openinterp.guard import FabricationGuard, FabricationGuardError

console = Console()


@click.group()
@click.version_option(__version__, prog_name="openinterp")
def main():
    """openinterp — operational mechanistic interpretability.

    Full docs: https://openinterp.org
    Notebooks: https://github.com/OpenInterpretability/notebooks
    """


# --- atlas -------------------------------------------------------------------

@main.command()
@click.argument("query")
@click.option("--model", "-m", default=None, help="Filter by HF model ID.")
@click.option("--limit", "-n", default=10, help="Max results (default 10).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def atlas(query: str, model: Optional[str], limit: int, as_json: bool):
    """Search the Atlas for features matching QUERY."""
    features = search_features(query, model=model, limit=limit)
    if as_json:
        console.print_json(
            json.dumps([f.model_dump() for f in features], ensure_ascii=False)
        )
        return
    if not features:
        console.print(f"[yellow]No features found for '{query}'[/yellow]")
        return
    t = Table(title=f"Atlas results: '{query}'")
    t.add_column("ID", style="cyan", no_wrap=True)
    t.add_column("Name", style="bold")
    t.add_column("Model", style="dim")
    t.add_column("AUROC", justify="right")
    t.add_column("Description")
    for f in features:
        auroc = f"{f.auroc:.2f}" if f.auroc is not None else "—"
        t.add_row(f.id, f.name, f.model, auroc, f.description[:70])
    console.print(t)


# --- trace -------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", required=True, help="HF model ID, e.g. 'google/gemma-2-2b'.")
@click.option("--sae-repo", required=True, help="HF SAE repo, e.g. 'YOUR/gemma2-2b-sae-first'.")
@click.option("--prompt", "-p", required=True, help="Input prompt.")
@click.option("--layer", "-l", required=True, type=int, help="Target residual-stream layer.")
@click.option("--d-model", default=2304, help="Base model hidden dim (default 2304 for Gemma-2-2B).")
@click.option("--d-sae", default=16384, help="SAE dictionary size.")
@click.option("--k", default=32, help="TopK sparsity.")
@click.option("--max-new-tokens", default=30, help="Tokens to generate.")
@click.option("--top-n", default=10, help="Top features to include in trace.")
@click.option("--device", default=None, help="cuda / cpu (auto if omitted).")
@click.option("--catalog", type=click.Path(exists=True), default=None,
              help="Optional feature_catalog.json from notebook 04 for labels.")
@click.option("--out", "-o", type=click.Path(), default="trace.json",
              help="Output file (default trace.json).")
def trace(
    model: str,
    sae_repo: str,
    prompt: str,
    layer: int,
    d_model: int,
    d_sae: int,
    k: int,
    max_new_tokens: int,
    top_n: int,
    device: Optional[str],
    catalog: Optional[str],
    out: str,
):
    """Generate a Trace (JSON) from model + SAE + prompt.

    Requires optional dependencies:  pip install 'openinterp\\[full]'

    Example:

        openinterp trace -m google/gemma-2-2b \\
                         --sae-repo YOUR/gemma2-2b-sae-first \\
                         -p "The capital of France is" \\
                         -l 12 --d-model 2304 --d-sae 16384 --k 64
    """
    catalog_data = None
    if catalog:
        catalog_data = json.loads(Path(catalog).read_text())

    try:
        with console.status("[bold magenta]Loading model and SAE…"):
            t = generate_trace(
                model_id=model,
                prompt=prompt,
                sae_repo=sae_repo,
                layer=layer,
                d_model=d_model,
                d_sae=d_sae,
                k=k,
                max_new_tokens=max_new_tokens,
                top_n_features=top_n,
                device=device,
                feature_catalog=catalog_data,
            )
    except TraceUnavailable as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    Path(out).write_text(t.model_dump_json(indent=2))
    console.print(f"[green]✓[/green] Trace saved to {out}")
    console.print(f"  prompt: {t.prompt[:80]}{'…' if len(t.prompt) > 80 else ''}")
    console.print(f"  tokens: {len(t.tokens)}")
    console.print(f"  features: {len(t.features)} "
                  f"(top: {', '.join(f.id for f in t.features[:5])}…)")
    console.print("\nView it at [cyan]https://openinterp.org/observatory/trace[/cyan] "
                  "(Q2 upload endpoint pending).")


# --- guard -------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", required=True, help="HF model ID (e.g. 'Qwen/Qwen3.6-27B').")
@click.option("--prompt", "-p", required=True, help="Input prompt to score / generate.")
@click.option("--mode", type=click.Choice(["detect", "warn", "abstain"]), default="detect",
              help="detect = score only;  warn = flag in output;  abstain = replace high-score with uncertainty response.")
@click.option("--threshold", type=float, default=None,
              help="Override calibrated threshold (default uses probe metadata).")
@click.option("--probe-repo", default=None,
              help="Override default probe registry — HF dataset with probe.joblib + meta.json.")
@click.option("--max-new-tokens", default=128, help="Tokens to generate when not abstaining.")
@click.option("--device", default=None, help="cuda / cpu (auto if omitted).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def guard(
    model: str,
    prompt: str,
    mode: str,
    threshold: Optional[float],
    probe_repo: Optional[str],
    max_new_tokens: int,
    device: Optional[str],
    as_json: bool,
):
    """Run FabricationGuard on a prompt — score + optional abstention.

    Requires optional dependencies:  pip install 'openinterp\\[full]'

    Example:

        openinterp guard -m Qwen/Qwen3.6-27B \\
                         -p "Who is Bambale Osby?" \\
                         --mode abstain
    """
    try:
        with console.status("[bold magenta]Loading model + probe…"):
            from transformers import AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText as _ModelCls
            except ImportError:
                from transformers import AutoModelForCausalLM as _ModelCls
            import torch

            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            kwargs = dict(
                dtype=torch.bfloat16, attn_implementation="sdpa",
                trust_remote_code=True,
            )
            if device:
                kwargs["device_map"] = {"": device}
            else:
                kwargs["device_map"] = "auto"
            try:
                hf_model = _ModelCls.from_pretrained(model, **kwargs)
            except Exception:
                from transformers import AutoModelForCausalLM
                hf_model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            hf_model.eval()

            g = FabricationGuard.from_pretrained(model, probe_repo=probe_repo,
                                                  threshold=threshold)
            g.attach(hf_model, tok)

        with console.status("[bold magenta]Scoring + generating…"):
            out = g.generate(prompt, mode=mode, max_new_tokens=max_new_tokens)

    except FabricationGuardError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if as_json:
        console.print_json(json.dumps(out, ensure_ascii=False))
        return

    score_color = "red" if out["flagged"] else ("yellow" if out["score"] > 0.4 else "green")
    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print(f"[bold]Mode:[/bold] {out['mode']}    [bold]Threshold:[/bold] {out['threshold']:.3f}")
    console.print(f"[bold]Score:[/bold] [{score_color}]{out['score']:.3f}[/{score_color}]    "
                  f"[bold]Flagged:[/bold] {'⚠  YES' if out['flagged'] else 'no'}")
    if out["abstained"]:
        console.print(f"[bold]Status:[/bold] [yellow]ABSTAINED[/yellow] (score above threshold)")
    console.print(f"\n[bold]Output:[/bold]")
    console.print(out["text"])


# --- info --------------------------------------------------------------------

@main.command()
def info():
    """Show installed version + optional dep status."""
    console.print(f"[bold]openinterp[/bold] v{__version__}")
    console.print(f"  homepage: [cyan]https://openinterp.org[/cyan]")
    console.print(f"  repo: [cyan]https://github.com/OpenInterpretability/cli[/cyan]")
    try:
        import torch
        import transformers
        console.print(f"  [green]✓ full stack[/green] — torch {torch.__version__}, "
                      f"transformers {transformers.__version__}")
    except ImportError:
        console.print(f"  [yellow]○ lite[/yellow] — install "
                      f"'openinterp\\[full]' for trace generation")


if __name__ == "__main__":
    main()
