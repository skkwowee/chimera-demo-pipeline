"""chimera-demo CLI — top-level orchestration.

Subcommands:
    scrape          — print N matches from HLTV (dry run, no download)
    fetch-match     — print one match's metadata + demo URL
    run             — full pipeline: scrape → download → extract → upload → manifest
    manifest        — show current state of the on-HF processed manifest

Designed to be runnable from a fresh RunPod worker:
    # one-shot batch
    chimera-demo run --stars 3 --max-matches 50 --repo skkwowee/chimera-cs2

    # resume — auto-skips already-processed (manifest on HF is source of truth)
    chimera-demo run --stars 3 --max-matches 50 --repo skkwowee/chimera-cs2
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import typer
from huggingface_hub import HfApi
from rich import print as rprint
from rich.table import Table

from .download import download_rar, extract_dems, normalize_demo_name, _new_scraper
from .hltv import HLTVScraper, MatchSummary
from .manifest import Manifest, ManifestEntry
from .upload import upload_demos

app = typer.Typer(help=__doc__.split("\n\n")[0], no_args_is_help=True)


@app.command()
def scrape(
    stars: int = typer.Option(3, help="HLTV match tier filter (1-5)"),
    max_matches: int = typer.Option(20, "--max-matches", "-n"),
    start_offset: int = typer.Option(0, help="paginate from offset N"),
):
    """Print N matches from HLTV listings (no downloads, no uploads)."""
    scraper = HLTVScraper()
    table = Table("id", "stars", "teams", "score", "event")
    n = 0
    for m in scraper.iter_matches(stars=stars, max_matches=max_matches,
                                    start_offset=start_offset):
        table.add_row(str(m.match_id), str(m.stars),
                       f"{m.team1} vs {m.team2}", m.score, m.event)
        n += 1
    rprint(table)
    rprint(f"[green]{n} matches scraped[/green]")


@app.command()
def fetch_match(
    match_id: int = typer.Argument(...),
    slug: str = typer.Argument(..., help="URL slug, e.g. 'furia-vs-vitality-blast-final'"),
):
    """Print one match's metadata including demo URL."""
    scraper = HLTVScraper()
    md = scraper.fetch_match(match_id, slug)
    rprint({
        "match_id": md.summary.match_id,
        "teams": f"{md.summary.team1} vs {md.summary.team2}",
        "event": md.summary.event,
        "score": md.summary.score,
        "maps": md.maps,
        "demo_url": md.demo_url,
        "demo_id": md.demo_id,
    })


@app.command()
def manifest(
    repo: str = typer.Option("skkwowee/chimera-cs2"),
    show_last: int = typer.Option(10, "--show-last", "-n"),
):
    """Show the on-HF processed manifest's most recent entries."""
    api = HfApi()
    mf = Manifest(api, repo)
    mf.load()
    rprint(f"[bold]Manifest:[/bold] {len(mf.entries)} matches on {repo}")
    if mf.entries:
        table = Table("match_id", "teams", "event", "score", "demos", "processed")
        for e in mf.entries[-show_last:]:
            table.add_row(str(e.match_id),
                          f"{e.team1} vs {e.team2}",
                          e.event[:30], e.score,
                          str(len(e.demo_files)),
                          time.strftime("%Y-%m-%d", time.gmtime(e.processed_at_unix)))
        rprint(table)


@app.command()
def run(
    repo: str = typer.Option("skkwowee/chimera-cs2"),
    stars: int = typer.Option(3, help="HLTV match tier (1-5; 5=LAN majors)"),
    max_matches: int = typer.Option(10, "--max-matches", "-n",
                                      help="upload at most N new matches this run"),
    start_offset: int = typer.Option(0),
    dry_run: bool = typer.Option(False, "--dry-run",
                                   help="scrape + download + extract + name; SKIP upload"),
):
    """Full pipeline: HLTV → demo .rar → .dem → HF dataset (with manifest update).

    Storage discipline: each match goes through `tempfile.TemporaryDirectory`,
    cleaned up after upload. Peak local disk = one match's files
    (~500MB-2GB depending on series length).
    """
    api = HfApi()
    rprint(f"[bold]Loading manifest from {repo}...[/bold]")
    mf = Manifest(api, repo)
    mf.load()
    rprint(f"  {len(mf.entries)} matches already processed")

    scraper = HLTVScraper()
    binary_scraper = _new_scraper()
    processed_this_run = 0
    skipped = 0
    failed: list[tuple[int, str]] = []

    rprint(f"[bold]Scraping HLTV results (stars={stars}, offset={start_offset})...[/bold]")
    for summary in scraper.iter_matches(stars=stars, max_matches=10_000,
                                          start_offset=start_offset):
        if processed_this_run >= max_matches:
            break
        if mf.has(summary.match_id):
            skipped += 1
            continue
        rprint(f"\n[cyan]→ match {summary.match_id}: {summary.team1} vs {summary.team2} "
               f"({summary.event}, {summary.score}, {summary.stars}★)[/cyan]")
        try:
            md = scraper.fetch_match(summary.match_id, summary.slug)
            if not md.demo_url:
                rprint(f"  [yellow]no demo URL; skipping[/yellow]")
                failed.append((summary.match_id, "no_demo_url"))
                continue
            with tempfile.TemporaryDirectory(prefix="chimera_demo_") as tmpd:
                tmp = Path(tmpd)
                rar = tmp / f"{summary.match_id}.rar"
                rprint(f"  downloading {md.demo_url} → {rar.name}")
                download_rar(md.demo_url, rar, scraper=binary_scraper)
                rprint(f"  extracting .dem files...")
                dems = extract_dems(rar, tmp)
                rprint(f"  {len(dems)} .dem files extracted")
                # Rename to canonical chimera form before upload
                renamed: list[Path] = []
                for i, dem in enumerate(dems, start=1):
                    new_name = normalize_demo_name(
                        dem, summary.team1, summary.team2,
                        summary.match_id, map_index=i,
                    )
                    new_path = dem.parent / new_name
                    dem.rename(new_path)
                    renamed.append(new_path)
                    rprint(f"    {dem.name} → {new_name}")
                bytes_total = sum(p.stat().st_size for p in renamed)
                if dry_run:
                    rprint(f"  [yellow][DRY] skip upload of {len(renamed)} files "
                           f"({bytes_total/1e6:.0f} MB)[/yellow]")
                else:
                    rprint(f"  uploading {len(renamed)} files ({bytes_total/1e6:.0f} MB)...")
                    repo_paths = upload_demos(
                        api, repo, renamed,
                        commit_message=(
                            f"Add {summary.team1} vs {summary.team2} ({summary.event}) "
                            f"— {len(renamed)} maps, match {summary.match_id}"
                        ),
                    )
                    entry = ManifestEntry(
                        match_id=summary.match_id, slug=summary.slug,
                        team1=summary.team1, team2=summary.team2,
                        event=summary.event, score=summary.score,
                        stars=summary.stars, date_unix=summary.date_unix,
                        demo_files=repo_paths, bytes_total=bytes_total,
                    )
                    mf.add(entry)
                    # Push manifest after EACH match so a crash doesn't lose the
                    # association. Single small file, cheap upload.
                    mf.push(commit_message=f"manifest: +match {summary.match_id}")
                # Count toward --max-matches regardless of dry-run vs real upload
                processed_this_run += 1
        except KeyboardInterrupt:
            rprint("[red]interrupted by user[/red]")
            sys.exit(130)
        except Exception as e:
            rprint(f"  [red]failed: {type(e).__name__}: {e}[/red]")
            failed.append((summary.match_id, f"{type(e).__name__}: {e}"))

    rprint(f"\n[bold green]Done.[/bold green] "
           f"processed={processed_this_run} skipped={skipped} failed={len(failed)}")
    if failed:
        rprint("[red]Failures:[/red]")
        for mid, why in failed:
            rprint(f"  {mid}: {why}")


if __name__ == "__main__":
    app()
