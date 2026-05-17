"""Tick-sequence build step: HF .dem -> chimera build_tick_sequences.py -> HF .pt.

Pipeline shape:
    demo manifest on HF  (processed_manifest.jsonl)
              |
              v
    pick a match not yet in tick-sequences manifest
              |
              v
    download its .dem files to a tempdir
              |
              v
    chimera/scripts/parse_demos.py  (per-tick parquet + per-event JSON)
              |
              v
    chimera/scripts/build_tick_sequences.py  (per-round .pt tensors)
              |
              v
    upload `train.pt` + `feature_schema_v1.json` + `manifest.json`
    to tick_sequences/<match_id>/ on HF
              |
              v
    append a line to processed_tick_sequences_manifest.jsonl on HF

Crash safety: each match is its own tempdir + its own atomic commit + its
own manifest line. A kill mid-batch leaves the previously-finished matches
fully consistent on HF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from rich import print as rprint

from .manifest import (
    Manifest,
    ManifestEntry,
    TickSequenceManifestEntry,
    TICK_SEQUENCES_MANIFEST_PATH,
)

# Enable parallel chunked uploads — mirrors upload.py
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _download_demos_for_match(
    api: HfApi, repo_id: str, entry: ManifestEntry, dest_dir: Path,
) -> list[Path]:
    """Pull every .dem listed in the manifest entry into dest_dir.

    Returns local paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for repo_path in entry.demo_files:
        # hf_hub_download caches to the HF cache dir; we copy into dest_dir
        # so the tempdir owns the lifecycle and we can move/rename freely.
        cached = api.hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
        )
        local = dest_dir / Path(repo_path).name
        shutil.copy(cached, local)
        local_paths.append(local)
    return local_paths


def _run_chimera_script(
    chimera_dir: Path, script_rel: str, args: list[str], cwd: Path,
    extra_env: dict | None = None,
) -> None:
    """Invoke a chimera script via chimera's venv python.

    Why subprocess: chimera deps (awpy/torch) live in chimera's venv and
    aren't installed into the pipeline venv. We don't want to make the
    pipeline package depend on those — it'd defeat the "fresh worker"
    runnability goal.

    Why we run the SANDBOX COPY of the script (cwd/script_rel) instead of
    the real one in chimera_dir: the chimera scripts compute
    `REPO = Path(__file__).resolve().parent.parent`. If we invoked
    chimera/scripts/parse_demos.py directly, REPO would be the real
    chimera repo and the script would read/write data/ there. We need
    REPO == sandbox, so we run the copy at `cwd/script_rel`.
    """
    chimera_py = chimera_dir / ".venv" / "bin" / "python"
    if not chimera_py.exists():
        raise FileNotFoundError(
            f"chimera venv python missing at {chimera_py}. "
            f"Activate chimera's env or pass --chimera-dir.")
    sandbox_script = cwd / script_rel
    cmd = [str(chimera_py), str(sandbox_script), *args]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    rprint(f"    [dim]$ {' '.join(cmd)}[/dim]")
    rprint(f"    [dim]  (cwd={cwd})[/dim]")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env,
                          capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            rprint(f"    [dim]| {line}[/dim]")
    if proc.returncode != 0:
        if proc.stderr:
            for line in proc.stderr.rstrip().splitlines():
                rprint(f"    [red]| {line}[/red]")
        raise RuntimeError(
            f"{script_rel} exited with code {proc.returncode}")


def _stage_chimera_sandbox(stage_root: Path, chimera_dir: Path) -> Path:
    """Build a minimal chimera-repo-shaped sandbox at `stage_root`.

    Layout (mirroring what the scripts expect):
        stage_root/scripts/parse_demos.py            (copy)
        stage_root/scripts/build_tick_sequences.py   (copy)
        stage_root/data/demos/                       (we'll drop .dem files here)
        stage_root/data/processed/                   (script outputs land here)

    Why COPY not symlink: both scripts compute
    `REPO = Path(__file__).resolve().parent.parent`. `resolve()` follows
    symlinks, so a symlinked script would set REPO to the real chimera repo
    and write outputs there instead of into the sandbox. Copying keeps
    __file__ inside the sandbox. We never modify the copied files.
    """
    (stage_root / "scripts").mkdir(parents=True, exist_ok=True)
    (stage_root / "data" / "demos").mkdir(parents=True, exist_ok=True)
    (stage_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for script in ("parse_demos.py", "build_tick_sequences.py"):
        src = chimera_dir / "scripts" / script
        dst = stage_root / "scripts" / script
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy(src, dst)
    return stage_root


def _upload_tick_sequence_artifacts(
    api: HfApi, repo_id: str, match_id: int,
    artifacts: list[Path], repo_prefix: str,
    commit_message: str,
) -> list[str]:
    """Atomically upload every artifact under tick_sequences/<match_id>/.

    Returns the list of path_in_repo strings.
    """
    ops: list[CommitOperationAdd] = []
    repo_paths: list[str] = []
    for p in artifacts:
        if not p.exists():
            raise FileNotFoundError(p)
        repo_path = f"{repo_prefix}/{p.name}"
        repo_paths.append(repo_path)
        ops.append(CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(p),
        ))
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=commit_message,
    )
    return repo_paths


def process_one(
    api: HfApi, repo: str, entry: ManifestEntry,
    chimera_dir: Path, downsample: int,
    dry_run: bool,
) -> TickSequenceManifestEntry | None:
    """Run the full process step for one match. Returns the manifest entry
    on success, or None if no usable output was produced.

    Cleanup: the surrounding `tempfile.TemporaryDirectory` context manager
    in the caller guarantees the tempdir is removed even on exception.
    """
    rprint(f"  [bold]download[/bold] {len(entry.demo_files)} .dem files...")
    with tempfile.TemporaryDirectory(prefix="chimera_tickseq_") as tmpd:
        tmp = Path(tmpd)
        sandbox = _stage_chimera_sandbox(tmp / "sandbox", chimera_dir)
        demos_dir = sandbox / "data" / "demos"
        local_dems = _download_demos_for_match(api, repo, entry, demos_dir)
        rprint(f"    {len(local_dems)} files in {demos_dir}")

        # --- parse_demos.py: .dem -> per-tick parquet + per-event JSON
        rprint(f"  [bold]parse_demos[/bold]...")
        _run_chimera_script(
            chimera_dir, "scripts/parse_demos.py",
            args=["--workers", "1"],   # serial — RAM headroom on small pods
            cwd=sandbox,
        )

        processed_dir = sandbox / "data" / "processed" / "demos"
        parqs = sorted(processed_dir.glob("*_ticks.parquet"))
        if not parqs:
            rprint(f"  [red]parse produced no parquets; skipping[/red]")
            return None

        # --- build_tick_sequences.py: parquet -> per-round .pt tensors
        # Put everything in `train.pt` (val=0) — this is per-match output,
        # not a global split. Downstream training does its own split.
        rprint(f"  [bold]build_tick_sequences[/bold] ({len(parqs)} demos)...")
        _run_chimera_script(
            chimera_dir, "scripts/build_tick_sequences.py",
            args=["--downsample", str(downsample), "--val-demos", "0"],
            cwd=sandbox,
        )

        ts_dir = sandbox / "data" / "processed" / "tick_sequences"
        train_pt = ts_dir / "train.pt"
        if not train_pt.exists() or train_pt.stat().st_size == 0:
            rprint(f"  [red]no train.pt produced; skipping[/red]")
            return None

        # Collect every artifact in the tick_sequences output directory so
        # downstream consumers get the schema + manifest alongside the .pt.
        # (val.pt won't exist when --val-demos 0.)
        artifacts = sorted(p for p in ts_dir.iterdir() if p.is_file())
        sizes_mb = {p.name: p.stat().st_size / 1e6 for p in artifacts}
        rprint(f"  [bold]artifacts[/bold]: " + ", ".join(
            f"{n} ({s:.1f} MB)" for n, s in sizes_mb.items()))

        # Pull stats out of train.pt header for the manifest line. Lazy import
        # so the pipeline venv isn't forced to have torch.
        # If chimera venv doesn't share site-packages with pipeline venv we
        # can't easily load the .pt here. So inspect via the chimera python.
        n_rounds, total_ticks, feature_dim = _summarize_pt(chimera_dir, train_pt)

        bytes_total = sum(p.stat().st_size for p in artifacts)
        repo_prefix = f"tick_sequences/{entry.match_id}"

        if dry_run:
            rprint(f"  [yellow][DRY] skip upload of {len(artifacts)} files "
                   f"({bytes_total/1e6:.1f} MB) -> {repo_prefix}/[/yellow]")
            return TickSequenceManifestEntry(
                match_id=entry.match_id, slug=entry.slug,
                team1=entry.team1, team2=entry.team2, event=entry.event,
                pt_files=[f"{repo_prefix}/{p.name}" for p in artifacts
                          if p.suffix == ".pt"],
                feature_schema_path=next(
                    (f"{repo_prefix}/{p.name}" for p in artifacts
                     if p.name.startswith("feature_schema")), None),
                n_rounds=n_rounds, total_ticks=total_ticks,
                feature_dim=feature_dim, downsample=downsample,
                bytes_total=bytes_total,
                source_demo_files=list(entry.demo_files),
            )

        rprint(f"  [bold]upload[/bold] -> {repo_prefix}/ ({bytes_total/1e6:.1f} MB)")
        repo_paths = _upload_tick_sequence_artifacts(
            api, repo, entry.match_id, artifacts, repo_prefix,
            commit_message=(
                f"Add tick_sequences for {entry.team1} vs {entry.team2} "
                f"({entry.event}) — {n_rounds} rounds, match {entry.match_id}"
            ),
        )
        return TickSequenceManifestEntry(
            match_id=entry.match_id, slug=entry.slug,
            team1=entry.team1, team2=entry.team2, event=entry.event,
            pt_files=[p for p in repo_paths if p.endswith(".pt")],
            feature_schema_path=next(
                (p for p in repo_paths if p.rsplit("/", 1)[-1]
                 .startswith("feature_schema")), None),
            n_rounds=n_rounds, total_ticks=total_ticks,
            feature_dim=feature_dim, downsample=downsample,
            bytes_total=bytes_total,
            source_demo_files=list(entry.demo_files),
        )


def _summarize_pt(chimera_dir: Path, pt_path: Path) -> tuple[int, int, int]:
    """Run the chimera python inline to read (n_rounds, total_ticks, feature_dim)
    from a saved tensor bundle. Returns zeros on any failure (best-effort).
    """
    chimera_py = chimera_dir / ".venv" / "bin" / "python"
    code = (
        "import torch, sys; "
        "d = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "n = len(d.get('tensors', [])); "
        "t = sum(int(x.shape[0]) for x in d.get('tensors', [])); "
        f"f = int(d.get('feature_dim', 0)); "
        "print(f'{n} {t} {f}')"
    )
    proc = subprocess.run(
        [str(chimera_py), "-c", code, str(pt_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return (0, 0, 0)
    try:
        n, t, f = proc.stdout.strip().split()
        return (int(n), int(t), int(f))
    except Exception:
        return (0, 0, 0)


def run_process(
    repo: str, chimera_dir: Path, max_matches: int,
    dry_run: bool, downsample: int,
) -> None:
    """Iterate the demo manifest, run tick-sequence build for each unprocessed
    match, push results match-by-match with crash-safe manifest discipline.
    """
    api = HfApi()
    rprint(f"[bold]Loading demo manifest from {repo}...[/bold]")
    demo_mf = Manifest(api, repo)
    demo_mf.load()
    rprint(f"  {len(demo_mf.entries)} demo matches on HF")

    rprint(f"[bold]Loading tick-sequence manifest...[/bold]")
    ts_mf = Manifest(
        api, repo,
        path=TICK_SEQUENCES_MANIFEST_PATH,
        entry_cls=TickSequenceManifestEntry,
    )
    ts_mf.load()
    rprint(f"  {len(ts_mf.entries)} matches already tick-sequenced")

    pending = [e for e in demo_mf.entries if not ts_mf.has(e.match_id)]
    rprint(f"[bold]Pending: {len(pending)} matches[/bold] "
           f"(cap this run: {max_matches})")

    processed_this_run = 0
    failed: list[tuple[int, str]] = []
    for entry in pending:
        if processed_this_run >= max_matches:
            break
        rprint(f"\n[cyan]→ match {entry.match_id}: {entry.team1} vs {entry.team2} "
               f"({entry.event})[/cyan]")
        t0 = time.time()
        try:
            new_entry = process_one(
                api, repo, entry, chimera_dir, downsample, dry_run,
            )
            if new_entry is None:
                failed.append((entry.match_id, "no_output"))
                continue
            if not dry_run:
                ts_mf.add(new_entry)
                ts_mf.push(commit_message=(
                    f"tick_sequences manifest: +match {entry.match_id}"
                ))
            processed_this_run += 1
            rprint(f"  [green]done[/green] in {time.time()-t0:.0f}s "
                   f"({new_entry.n_rounds} rounds, "
                   f"{new_entry.total_ticks:,} ticks, "
                   f"dim={new_entry.feature_dim})")
        except KeyboardInterrupt:
            rprint("[red]interrupted by user[/red]")
            sys.exit(130)
        except Exception as e:
            rprint(f"  [red]failed: {type(e).__name__}: {e}[/red]")
            failed.append((entry.match_id, f"{type(e).__name__}: {e}"))

    rprint(f"\n[bold green]Done.[/bold green] "
           f"processed={processed_this_run} failed={len(failed)} "
           f"pending_remaining={max(0, len(pending) - processed_this_run)}")
    if failed:
        rprint("[red]Failures:[/red]")
        for mid, why in failed:
            rprint(f"  {mid}: {why}")
