"""openinterp — command line interface."""
import click
from rich.console import Console
from rich.table import Table
from openinterp import __version__, search_features, generate_trace

console = Console()


@click.group()
@click.version_option(__version__, prog_name="openinterp")
def main():
    """openinterp — search Atlas, generate Traces, upload SAEs.

    Full docs: https://openinterp.org/docs
    """
    pass


@main.command()
@click.argument("query")
@click.option("--model", "-m", default=None, help="Filter by HF model ID")
@click.option("--limit", "-n", default=10, help="Max results")
def atlas(query: str, model: str | None, limit: int):
    """Search the Atlas for features matching QUERY."""
    features = search_features(query, model=model, limit=limit)
    if not features:
        console.print(f"[yellow]No features found for '{query}'[/yellow]")
        return
    t = Table(title=f"Atlas results: '{query}'")
    t.add_column("ID", style="cyan")
    t.add_column("Name", style="bold")
    t.add_column("Model", style="dim")
    t.add_column("AUROC", justify="right")
    t.add_column("Description")
    for f in features:
        t.add_row(f.id, f.name, f.model, f"{f.auroc:.2f}" if f.auroc else "—", f.description[:60])
    console.print(t)


@main.command()
@click.option("--model", "-m", required=True, help="HF model ID, e.g. Qwen/Qwen3.6-27B")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--sae-repo", default=None, help="HF SAE repo, auto-detected if omitted")
def trace(model: str, prompt: str, sae_repo: str | None):
    """Generate a feature-activation Trace for PROMPT on MODEL."""
    try:
        t = generate_trace(model, prompt, sae_repo=sae_repo)
        console.print_json(t.model_dump_json(indent=2))
    except NotImplementedError as e:
        console.print(f"[yellow]{e}[/yellow]")


if __name__ == "__main__":
    main()
