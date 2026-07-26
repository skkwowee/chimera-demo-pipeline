"""chimera-demo CLI — top-level orchestration.

Subcommands:
    scrape          — print N matches from HLTV (dry run, no download)
    fetch-match     — print one match's metadata + demo URL
    run             — stage 1 ingest: scrape → download → extract → upload → manifest
    process         — stage 2: HF .dem → tick-sequence tensors on HF
    manifest        — show the on-HF processed manifest
    failures        — list persistent failure records (failures.jsonl)
    backfill        — push legacy local-only demos to HF

Runnable from a fresh RunPod worker; reruns auto-skip already-processed
matches (the manifest on HF is the source of truth):
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

from .download import (download_rar, extract_dems, normalize_demo_name,
                       order_dems_for_series, detect_map_name, _new_scraper)
from .hltv import HLTVScraper, MatchSummary, ScrapeLayoutError
from .manifest import (Manifest, ManifestEntry, FailureEntry,
                       FAILURES_MANIFEST_PATH, record_failure,
                       should_skip_failed)
from .process import run_process
from .upload import upload_demos

# NOTE: the commentary commands (find-vods, transcribe) and their modules
# (vod, transcribe, commentary_pilot, demo_anchors) are PARKED in
# parked/commentary/ until the language-grounding phase. See that dir's README.

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
    team: list[str] = typer.Option(
        None, "--team",
        help="case-insensitive substring filter on team names (repeatable; "
             "matches if EITHER team name contains ANY of the provided substrings). "
             "Example: --team vitality --team spirit"),
    top_teams: bool = typer.Option(
        False, "--top-teams",
        help="shortcut for the current HLTV top ~10 teams (vitality, spirit, "
             "falcons, mouz, natus vincere, faze, g2, mongolz, astralis, liquid, "
             "aurora, paris legion). Overrides --team if both passed."),
    event: list[str] = typer.Option(
        None, "--event",
        help="case-insensitive substring filter on event name (repeatable; "
             "matches if event contains ANY)"),
    scan_limit: int = typer.Option(
        5000, "--scan-limit",
        help="max HLTV listings to walk before giving up on filters. Bumps if "
             "your --team filter is narrow and most listings don't match."),
    retry_failed: bool = typer.Option(
        False, "--retry-failed",
        help="retry matches with >=3 recorded failures in failures.jsonl"),
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

    fail_mf = Manifest(api, repo, path=FAILURES_MANIFEST_PATH,
                       entry_cls=FailureEntry)
    fail_mf.load()
    if fail_mf.entries:
        rprint(f"  {len(fail_mf.entries)} matches with recorded failures")

    scraper = HLTVScraper()
    binary_scraper = _new_scraper()
    processed_this_run = 0
    skipped = 0
    skipped_failed = 0
    failed: list[tuple[int, str]] = []

    TOP_TEAMS = [
        "vitality", "spirit", "falcons", "mouz", "natus vincere", "faze",
        "g2", "mongolz", "astralis", "liquid", "aurora", "paris legion",
        "furia", "navi",
    ]
    if top_teams:
        team = TOP_TEAMS
    team_filters = [t.lower() for t in (team or [])]
    event_filters = [e.lower() for e in (event or [])]
    if team_filters:
        rprint(f"[bold]Team filter (any):[/bold] {team_filters}")
    if event_filters:
        rprint(f"[bold]Event filter (any):[/bold] {event_filters}")

    def matches_filters(s) -> bool:
        if team_filters:
            t1 = s.team1.lower(); t2 = s.team2.lower()
            if not any(t in t1 or t in t2 for t in team_filters):
                return False
        if event_filters:
            ev = s.event.lower()
            if not any(e in ev for e in event_filters):
                return False
        return True

    rprint(f"[bold]Scraping HLTV results (stars={stars}, offset={start_offset}, "
           f"scan_limit={scan_limit})...[/bold]")
    scanned = 0
    filtered_out = 0
    for summary in scraper.iter_matches(stars=stars, max_matches=scan_limit,
                                          start_offset=start_offset):
        scanned += 1
        if processed_this_run >= max_matches:
            break
        if mf.has(summary.match_id):
            skipped += 1
            continue
        if not matches_filters(summary):
            filtered_out += 1
            continue
        if should_skip_failed(fail_mf, summary.match_id, retry_failed):
            skipped_failed += 1
            continue
        rprint(f"\n[cyan]→ match {summary.match_id}: {summary.team1} vs {summary.team2} "
               f"({summary.event}, {summary.score}, {summary.stars}★)[/cyan]")
        stage = "scrape"
        try:
            md = scraper.fetch_match(summary.match_id, summary.slug)
            if not md.demo_url:
                # _parse_match verified the demo SECTION exists (else it
                # raises ScrapeLayoutError) — this is 'not released yet'.
                rprint(f"  [yellow]no demo URL (not released yet?); skipping[/yellow]")
                failed.append((summary.match_id, "no_demo_url"))
                if not dry_run:
                    record_failure(fail_mf, summary.match_id, "scrape",
                                   "no_demo_url")
                    fail_mf.push(commit_message=(
                        f"failures: match {summary.match_id}"))
                continue
            with tempfile.TemporaryDirectory(prefix="chimera_demo_") as tmpd:
                tmp = Path(tmpd)
                rar = tmp / f"{summary.match_id}.rar"
                rprint(f"  downloading {md.demo_url} → {rar.name}")
                stage = "download"
                download_rar(md.demo_url, rar, scraper=binary_scraper)
                rprint(f"  extracting .dem files...")
                stage = "extract"
                dems = extract_dems(rar, tmp)
                rprint(f"  {len(dems)} .dem files extracted")
                # Series order: filenames with mN tokens sort correctly;
                # old-format names are ordered by the match page's map list.
                dems = order_dems_for_series(dems, md.maps)
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
                    stage = "upload"
                    rprint(f"  uploading {len(renamed)} files ({bytes_total/1e6:.0f} MB)...")
                    # Namespaced by match_id — a rematch of the same teams on
                    # the same map slot must not overwrite the earlier .dem.
                    repo_paths = upload_demos(
                        api, repo, renamed,
                        commit_message=(
                            f"Add {summary.team1} vs {summary.team2} ({summary.event}) "
                            f"— {len(renamed)} maps, match {summary.match_id}"
                        ),
                        match_id=summary.match_id,
                    )
                    entry = ManifestEntry(
                        match_id=summary.match_id, slug=summary.slug,
                        team1=summary.team1, team2=summary.team2,
                        event=summary.event, score=summary.score,
                        stars=summary.stars, date_unix=summary.date_unix,
                        demo_files=repo_paths, bytes_total=bytes_total,
                        demo_url=md.demo_url, demo_id=md.demo_id,
                        hltv_maps=list(md.maps),
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
        except ScrapeLayoutError:
            # Layout drift is NOT a per-match failure — abort loudly.
            raise
        except Exception as e:
            rprint(f"  [red]failed [{stage}]: {type(e).__name__}: {e}[/red]")
            failed.append((summary.match_id, f"{type(e).__name__}: {e}"))
            if not dry_run:
                try:
                    record_failure(fail_mf, summary.match_id, stage,
                                   f"{type(e).__name__}: {e}"[:500])
                    fail_mf.push(commit_message=(
                        f"failures: match {summary.match_id}"))
                except Exception as pe:
                    rprint(f"  [red]could not persist failure: {pe}[/red]")

    rprint(f"\n[bold green]Done.[/bold green] "
           f"processed={processed_this_run} skipped={skipped} "
           f"skipped_failed={skipped_failed} "
           f"filtered_out={filtered_out} scanned={scanned} failed={len(failed)}")
    if failed:
        rprint("[red]Failures:[/red]")
        for mid, why in failed:
            rprint(f"  {mid}: {why}")


@app.command()
def process(
    repo: str = typer.Option("skkwowee/chimera-cs2"),
    max_matches: int = typer.Option(5, "--max-matches", "-n",
                                      help="process at most N new matches this run"),
    chimera_dir: Path = typer.Option(
        Path("/home/soone/chimera"), "--chimera-dir",
        help="path to the chimera repo (contains scripts/ and .venv/)"),
    downsample: int = typer.Option(
        8, help="tick downsample factor for build_tick_sequences (8 = 64Hz→8Hz)"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                   help="download + parse + build; SKIP upload of .pt"),
    archive_parsed: bool = typer.Option(
        True, "--archive-parsed/--no-archive-parsed",
        help="also upload the parse intermediates (parquet + event JSONs, "
             "~32 MB/match) to parsed/<match_id>/ so future schema changes "
             "re-run only the build step, not the awpy parse"),
    from_parsed: bool = typer.Option(
        False, "--from-parsed",
        help="rebake path: download parsed/<match_id>/ instead of demos/ and "
             "skip the awpy parse (falls back to demos when no archive)"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed",
        help="retry matches with >=3 recorded failures in failures.jsonl"),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty",
        help="proceed even when chimera's builder scripts have uncommitted "
             "edits (bakes become unreproducible)"),
):
    """Tick-sequence build step: HF .dem → chimera tick-sequence tensors → HF .pt.

    For each match in the demo manifest not yet in the tick-sequences
    manifest — or baked under an OLD schema_version (re-bake):
      1. download its .dem files into a tempdir
      2. run chimera's parse_demos.py (→ per-tick parquet + per-event JSON)
      3. archive the parse bundle to parsed/<match_id>/ (default on)
      4. run chimera's build_tick_sequences.py (→ per-round .pt tensors)
      5. upload outputs under tick_sequences/<match_id>/ on HF
      6. append a line to processed_tick_sequences_manifest.jsonl
         (with schema_version + builder/pipeline commits + parser versions)

    Match-atomic: per-match tempdir + per-match commit + per-match manifest
    push, mirroring `run`'s crash-safety discipline.
    """
    run_process(
        repo=repo, chimera_dir=chimera_dir,
        max_matches=max_matches, dry_run=dry_run, downsample=downsample,
        archive_parsed=archive_parsed, from_parsed=from_parsed,
        retry_failed=retry_failed, allow_dirty=allow_dirty,
    )


@app.command()
def failures(
    repo: str = typer.Option("skkwowee/chimera-cs2"),
):
    """List the persistent failure records (failures.jsonl on HF)."""
    api = HfApi()
    fail_mf = Manifest(api, repo, path=FAILURES_MANIFEST_PATH,
                       entry_cls=FailureEntry)
    fail_mf.load()
    rprint(f"[bold]Failures:[/bold] {len(fail_mf.entries)} matches on {repo}")
    if fail_mf.entries:
        table = Table("match_id", "stage", "attempts", "last_attempt", "reason")
        for e in fail_mf.entries:
            table.add_row(str(e.match_id), e.stage, str(e.attempts),
                          time.strftime("%Y-%m-%d %H:%M",
                                        time.gmtime(e.last_attempt_unix)),
                          e.reason[:80])
        rprint(table)


@app.command()
def backfill(
    demos_dir: Path = typer.Option(..., "--demos-dir",
                                     help="local dir with canonical *.dem files"),
    repo: str = typer.Option("skkwowee/chimera-cs2"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                   help="print the missing-stem diff; no uploads"),
):
    """Backfill legacy LOCAL-ONLY demos to HF.

    HF is canonical but NOT a superset of the old local corpus (~54 of 81
    local demo stems have a single copy on one WSL2 disk). This scans
    --demos-dir, diffs against every .dem already on HF (flat or
    namespaced), groups missing stems by team pair, uploads each group, and
    appends synthetic manifest entries (deterministic negative match_id,
    event='local_backfill') so run/process skip-logic keys on match_id
    without colliding with HLTV ids.
    """
    import hashlib
    import re as _re

    local_dems = sorted(demos_dir.glob("*.dem"))
    if not local_dems:
        rprint(f"[red]no .dem files in {demos_dir}[/red]")
        raise typer.Exit(1)

    api = HfApi()
    rprint(f"[bold]Listing demos on {repo}...[/bold]")
    on_hf = {p.rsplit("/", 1)[-1]
             for p in api.list_repo_files(repo, repo_type="dataset")
             if p.startswith("demos/") and p.endswith(".dem")}
    rprint(f"  {len(on_hf)} .dem files on HF; {len(local_dems)} local")

    missing = [p for p in local_dems if p.name not in on_hf]
    rprint(f"[bold]Missing from HF: {len(missing)}[/bold]")
    if not missing:
        return

    # Group by team-pair token so each pseudo-match uploads atomically
    def group_key(stem: str) -> str:
        return _re.sub(r"-m\d+-\w+$", "", stem.lower())

    groups: dict[str, list[Path]] = {}
    for p in missing:
        groups.setdefault(group_key(p.stem), []).append(p)

    mf = Manifest(api, repo)
    mf.load()
    for key in sorted(groups):
        paths = groups[key]
        # Deterministic negative id — never collides with HLTV's positive ids
        synth_id = -int(hashlib.sha1(key.encode()).hexdigest()[:12], 16)
        names = [p.name for p in paths]
        if mf.has(synth_id):
            rprint(f"  [yellow]skip {key} (already backfilled)[/yellow]")
            continue
        bytes_total = sum(p.stat().st_size for p in paths)
        if dry_run:
            rprint(f"  [yellow][DRY] {key} (match_id={synth_id}): "
                   f"{names} ({bytes_total/1e6:.0f} MB)[/yellow]")
            continue
        rprint(f"  uploading {key} → demos/ ({len(paths)} files, "
               f"{bytes_total/1e6:.0f} MB)")
        repo_paths = upload_demos(
            api, repo, paths,
            commit_message=f"Backfill local-only demos: {key}",
        )
        t1, _, t2 = key.partition("-vs-")
        mf.add(ManifestEntry(
            match_id=synth_id, slug=f"local-{key}",
            team1=t1, team2=t2 or "", event="local_backfill",
            score="", stars=0, date_unix=None,
            demo_files=repo_paths, bytes_total=bytes_total,
        ))
        mf.push(commit_message=f"manifest: +backfill {key}")
    rprint("[bold green]Backfill done.[/bold green]")


if __name__ == "__main__":
    app()
