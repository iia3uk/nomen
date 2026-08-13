"""Typer CLI for the Nomen brand discovery engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from nomen import __version__
from nomen.config import Secrets, load_config
from nomen.pipeline.orchestrator import install_uvloop, run_engine

app = typer.Typer(
    name="nomen",
    help="Production-grade brand discovery engine for software products.",
    add_completion=False,
    no_args_is_help=False,
)
console = Console(legacy_windows=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    generate_batch: Optional[int] = typer.Option(None, "--generate-batch", "--generate"),
    online_batch: Optional[int] = typer.Option(None, "--online-batch", "--check"),
    target_clean: Optional[int] = typer.Option(None, "--target-clean", "--stop-after"),
    min_overall_score: Optional[float] = typer.Option(None, "--min-score"),
    min_beauty_score: Optional[float] = typer.Option(None, "--min-beauty", help="BeautyScore floor"),
    seed: Optional[str] = typer.Option(None, "--seed"),
    output_dir: Optional[str] = typer.Option(None, "--output"),
    strict: Optional[bool] = typer.Option(None, "--strict/--no-strict"),
    resume: Optional[bool] = typer.Option(None, "--resume/--fresh"),
    embeddings: Optional[bool] = typer.Option(None, "--embeddings/--no-embeddings"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Parallel HTTP checks"),
    workers: Optional[int] = typer.Option(None, "--workers", help="CPU process pool for generators (0=auto)"),
    version: Optional[bool] = typer.Option(None, "--version", is_eager=True),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit(0)
    if ctx.invoked_subcommand is not None:
        return

    overrides: dict = {}
    if generate_batch is not None:
        overrides["generate_batch"] = generate_batch
    if online_batch is not None:
        overrides["online_batch"] = online_batch
    if target_clean is not None:
        overrides["target_clean"] = target_clean
    if min_overall_score is not None:
        overrides["min_overall_score"] = min_overall_score
    if min_beauty_score is not None:
        overrides["min_beauty_score"] = min_beauty_score
    if seed is not None:
        overrides["seed"] = seed
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if strict is not None:
        overrides["strict"] = strict
    if resume is not None:
        overrides["resume"] = resume
    if concurrency is not None:
        overrides["concurrency"] = concurrency
    if workers is not None:
        overrides["workers"] = workers
    if embeddings is not None:
        overrides["embeddings"] = {"enabled": embeddings}

    cfg = load_config(config, overrides)
    secrets = Secrets()

    console.print(f"[bold]Nomen[/] v{__version__} — beauty-led brand hunt")
    console.print(
        f"batch={cfg.generate_batch:,} online={cfg.online_batch} "
        f"target={cfg.target_clean} min_score={cfg.min_overall_score} "
        f"min_beauty={cfg.min_beauty_score} "
        f"strict={cfg.strict} concurrency={cfg.concurrency} workers={cfg.workers or 'auto'}"
    )
    if not secrets.github_token:
        console.print("[yellow]GITHUB_TOKEN missing — GitHub may rate-limit.[/]")
    if not secrets.has_web_search:
        console.print("[yellow]No Brave/SerpAPI/Bing key — web/trademark depth reduced.[/]")

    install_uvloop()
    try:
        asyncio.run(run_engine(cfg, secrets))
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted — checkpoint saved. Re-run with --resume.[/]")
        raise typer.Exit(130) from None


@app.command("benchmark")
def benchmark(
    n: int = typer.Option(3000, "--n", help="Candidates to generate for diversity benchmark"),
    seed: str = typer.Option("benchmark", "--seed"),
) -> None:
    """Offline novelty/diversity benchmark (no network)."""
    from collections import Counter
    from statistics import mean

    from nomen.config import SimilarityConfig
    from nomen.diversity.novelty import NoveltyArchive
    from nomen.diversity.selector import DiversitySelector
    from nomen.filters.offline import OfflineFilterBank
    from nomen.generation.engine import GenerationEngine
    from nomen.scoring.scorer import score_candidate
    from nomen.similarity.engine import SimilarityEngine

    eng = GenerationEngine(seed)
    prov = eng.generate_all(n)
    off = OfflineFilterBank()
    ok, bad = off.apply(prov)
    sim = SimilarityEngine(SimilarityConfig())
    ok2, bad2 = sim.apply(ok)
    for c in ok2:
        score_candidate(c, eng.lm)
    sel = DiversitySelector(NoveltyArchive())
    result = sel.select(ok2, limit=100, winners=[])
    top = result.selected
    roots = Counter(c.scores.phonetic_root for c in top)
    gens = Counter(c.generator for c in top)
    console.print(f"Generated: {len(prov)}")
    console.print(f"Offline rejected: {len(bad)}")
    console.print(f"Similarity rejected: {len(bad2)}")
    console.print(f"Diversity Top100: {len(top)} (clusters rejected {len(result.rejected)})")
    if top:
        console.print(f"Unique roots: {len(roots)}  unique generators: {len(gens)}")
        console.print(f"Mean novelty: {mean(c.scores.novelty_score for c in top):.2f}")
        console.print(f"Mean brand:   {mean(c.scores.brand_score for c in top):.2f}")
        console.print(f"Max root share: {roots.most_common(1)[0]}")
        console.print("Top 20 (must look independent):")
        for c in top[:20]:
            console.print(
                f"  {c.display_name:12} nov={c.scores.novelty_score:5.1f} "
                f"brand={c.scores.brand_score:5.1f} root={c.scores.phonetic_root:6} "
                f"{c.scores.cv_pattern:8} {c.generator}"
            )


@app.command("version")
def show_version() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()
