"""Tick-sequence build step: HF .dem -> chimera build_tick_sequences.py -> HF .pt.

Pipeline shape:
    demo manifest on HF  (processed_manifest.jsonl)
              |
              v
    pick a match not yet in tick-sequences manifest
    (or baked under an OLD schema_version -> re-bake)
              |
              v
    download its .dem files to a tempdir
    (or its archived parsed/<match_id>/ bundle with --from-parsed)
              |
              v
    chimera/scripts/parse_demos.py  (per-tick parquet + per-event JSON)
              |
              v
    archive the parse bundle to parsed/<match_id>/ on HF  (--archive-parsed,
    default ON — future schema changes re-run ONLY the ~1min/match build step
    instead of the 3-6min/demo awpy parse or a full re-scrape)
              |
              v
    chimera/scripts/build_tick_sequences.py  (per-round .pt tensors)
              |
              v
    upload `train.pt` + `feature_schema*.json` + `manifest.json`
    to tick_sequences/<match_id>/ on HF
              |
              v
    append a line to processed_tick_sequences_manifest.jsonl on HF
    (with schema_version + builder/pipeline commit + parser versions)

Crash safety: each match is its own tempdir + its own atomic commit + its
own manifest line. A kill mid-batch leaves the previously-finished matches
fully consistent on HF.
"""

from __future__ import annotations

import json
import os
import re
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
    FAILURES_MANIFEST_PATH,
    FailureEntry,
    record_failure,
    should_skip_failed,
)

# Enable parallel chunked uploads — mirrors upload.py
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# demoparser2 < 0.41.3 hits EntityNotFound on Major demos — fail fast
# before downloading anything (awpy's own floor is only 0.38.0).
MIN_DEMOPARSER2 = (0, 41, 3)

# The builder must contain the site-from-plant-position fix (datasheet
# D-defect: awpy's round-level bomb_site label is broken corpus-wide).
BUILDER_FIX_MARKER = "derive_site_from_plant"


class StageFailure(RuntimeError):
    """A failure attributable to a specific pipeline stage (for failures.jsonl)."""

    def __init__(self, stage: str, msg: str):
        super().__init__(msg)
        self.stage = stage


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _git_head(repo_dir: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty(repo_dir: Path, paths: list[str]) -> bool | None:
    try:
        r = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain",
                            "--", *paths],
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


def _chimera_env_versions(chimera_dir: Path) -> tuple[str | None, str | None]:
    """(awpy_version, demoparser2_version) from chimera's venv. Aborts the
    run if demoparser2 is below MIN_DEMOPARSER2."""
    chimera_py = chimera_dir / ".venv" / "bin" / "python"
    code = ("import importlib.metadata as md; "
            "print(md.version('awpy')); print(md.version('demoparser2'))")
    proc = subprocess.run([str(chimera_py), "-c", code],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"cannot read awpy/demoparser2 versions from {chimera_py}: "
            f"{proc.stderr.strip()[-300:]}")
    awpy_v, demoparser_v = proc.stdout.strip().splitlines()[:2]
    if _vtuple(demoparser_v) < MIN_DEMOPARSER2:
        raise SystemExit(
            f"demoparser2 {demoparser_v} < "
            f"{'.'.join(map(str, MIN_DEMOPARSER2))} in chimera venv — "
            f"EntityNotFound on Major demos. Upgrade before processing "
            f"(pip install -U 'demoparser2>=0.41.3').")
    return awpy_v, demoparser_v


def _builder_schema_version(chimera_dir: Path) -> str:
    """Read SCHEMA_VERSION out of the builder script; hard-fail if the
    script predates the site-from-plant-position fix."""
    script = chimera_dir / "scripts" / "build_tick_sequences.py"
    text = script.read_text()
    if BUILDER_FIX_MARKER not in text:
        raise SystemExit(
            f"{script} predates the site-from-plant-position fix "
            f"(no `{BUILDER_FIX_MARKER}`) — it would bake the corpus-wide "
            f"broken awpy bomb_site label into fresh tensors "
            f"(chimera docs/datasheet.md). Update the chimera checkout.")
    m = re.search(r'^SCHEMA_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise SystemExit(f"{script} has no SCHEMA_VERSION constant")
    return m.group(1)


def _read_schema_version(ts_dir: Path) -> str | None:
    """The 'version' key of the schema JSON the builder just wrote."""
    for name in ("feature_schema.json", "feature_schema_v1.json"):
        p = ts_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text()).get("version")
            except Exception:
                return None
    return None


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


def _download_parsed_for_match(
    api: HfApi, repo_id: str, parsed_files: list[str], dest_dir: Path,
) -> list[Path]:
    """Pull an archived parsed/<match_id>/ bundle into dest_dir (rebake path)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for repo_path in parsed_files:
        cached = api.hf_hub_download(
            repo_id=repo_id, filename=repo_path, repo_type="dataset",
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


def _upload_artifacts(
    api: HfApi, repo_id: str,
    artifacts: list[Path], repo_prefix: str,
    commit_message: str,
) -> list[str]:
    """Atomically upload every artifact under `repo_prefix`/ in ONE commit.

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


def _collect_parse_bundle(processed_dir: Path) -> list[Path]:
    """Every per-demo parse output worth archiving: {stem}_ticks.parquet +
    the 5 event/header JSONs. ~12.8 MB/map measured — 1-3% of .dem size."""
    out = sorted(processed_dir.glob("*_ticks.parquet"))
    for suffix in ("kills", "bomb", "damages", "rounds", "header"):
        out.extend(sorted(processed_dir.glob(f"*_{suffix}.json")))
    return out


def process_one(
    api: HfApi, repo: str, entry: ManifestEntry,
    chimera_dir: Path, downsample: int,
    dry_run: bool,
    prov: dict | None = None,
    archive_parsed: bool = True,
    from_parsed: bool = False,
    prev_entry: TickSequenceManifestEntry | None = None,
) -> TickSequenceManifestEntry | None:
    """Run the full process step for one match. Returns the manifest entry
    on success, or None if no usable output was produced.

    prov: provenance computed once by run_process (schema_version,
    builder_commit, pipeline_commit, awpy/demoparser2 versions).
    from_parsed: rebake path — download the archived parsed/<match_id>/
    bundle instead of demos and skip the awpy parse (falls back to demos
    when no archive exists).

    Cleanup: the surrounding `tempfile.TemporaryDirectory` context manager
    in the caller guarantees the tempdir is removed even on exception.
    """
    prov = prov or {}
    with tempfile.TemporaryDirectory(prefix="chimera_tickseq_") as tmpd:
        tmp = Path(tmpd)
        sandbox = _stage_chimera_sandbox(tmp / "sandbox", chimera_dir)
        processed_dir = sandbox / "data" / "processed" / "demos"

        use_parsed = (from_parsed and prev_entry is not None
                      and prev_entry.parsed_files)
        parsed_repo_paths: list[str] = []
        parsed_bytes = 0
        if use_parsed:
            rprint(f"  [bold]download parsed[/bold] "
                   f"{len(prev_entry.parsed_files)} archived files "
                   f"(skipping demos + parse)...")
            try:
                locals_ = _download_parsed_for_match(
                    api, repo, prev_entry.parsed_files, processed_dir)
            except Exception as e:
                raise StageFailure("download", f"parsed bundle: {e}") from e
            parsed_repo_paths = list(prev_entry.parsed_files)
            parsed_bytes = sum(p.stat().st_size for p in locals_)
        else:
            rprint(f"  [bold]download[/bold] {len(entry.demo_files)} .dem files...")
            demos_dir = sandbox / "data" / "demos"
            try:
                local_dems = _download_demos_for_match(api, repo, entry, demos_dir)
            except Exception as e:
                raise StageFailure("download", str(e)) from e
            rprint(f"    {len(local_dems)} files in {demos_dir}")

            # --- parse_demos.py: .dem -> per-tick parquet + per-event JSON
            rprint(f"  [bold]parse_demos[/bold]...")
            try:
                _run_chimera_script(
                    chimera_dir, "scripts/parse_demos.py",
                    args=["--workers", "1"],   # serial — RAM headroom on small pods
                    cwd=sandbox,
                )
            except Exception as e:
                raise StageFailure("parse", str(e)) from e

        parqs = sorted(processed_dir.glob("*_ticks.parquet"))
        if not parqs:
            rprint(f"  [red]parse produced no parquets; skipping[/red]")
            return None

        # --- archive the parse bundle to parsed/<match_id>/ BEFORE build.
        # This is the corpus-dossier decision: keep intermediates so schema
        # changes cost a ~1min rebuild, not a re-parse or a re-scrape.
        if not use_parsed:
            bundle = _collect_parse_bundle(processed_dir)
            parsed_bytes = sum(p.stat().st_size for p in bundle)
            parsed_prefix = f"parsed/{entry.match_id}"
            if archive_parsed:
                if dry_run:
                    rprint(f"  [yellow][DRY] skip archive of {len(bundle)} "
                           f"parsed files ({parsed_bytes/1e6:.1f} MB) -> "
                           f"{parsed_prefix}/[/yellow]")
                    parsed_repo_paths = [f"{parsed_prefix}/{p.name}"
                                         for p in bundle]
                else:
                    rprint(f"  [bold]archive parsed[/bold] -> {parsed_prefix}/ "
                           f"({parsed_bytes/1e6:.1f} MB)")
                    try:
                        parsed_repo_paths = _upload_artifacts(
                            api, repo, bundle, parsed_prefix,
                            commit_message=(
                                f"Archive parsed intermediates for "
                                f"{entry.team1} vs {entry.team2} "
                                f"— match {entry.match_id}"),
                        )
                    except Exception as e:
                        raise StageFailure("upload", f"parsed archive: {e}") from e
            else:
                parsed_bytes = 0

        # --- build_tick_sequences.py: parquet -> per-round .pt tensors
        # Put everything in `train.pt` (val=0) — this is per-match output,
        # not a global split. Downstream training does its own split.
        rprint(f"  [bold]build_tick_sequences[/bold] ({len(parqs)} demos)...")
        try:
            _run_chimera_script(
                chimera_dir, "scripts/build_tick_sequences.py",
                args=["--downsample", str(downsample), "--val-demos", "0"],
                cwd=sandbox,
            )
        except Exception as e:
            raise StageFailure("bake", str(e)) from e

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

        schema_version = _read_schema_version(ts_dir) or prov.get("schema_version")
        bytes_total = sum(p.stat().st_size for p in artifacts)
        repo_prefix = f"tick_sequences/{entry.match_id}"

        def _mk_entry(pt_files: list[str], schema_path: str | None):
            return TickSequenceManifestEntry(
                match_id=entry.match_id, slug=entry.slug,
                team1=entry.team1, team2=entry.team2, event=entry.event,
                pt_files=pt_files,
                feature_schema_path=schema_path,
                n_rounds=n_rounds, total_ticks=total_ticks,
                feature_dim=feature_dim, downsample=downsample,
                bytes_total=bytes_total,
                source_demo_files=list(entry.demo_files),
                schema_version=schema_version,
                builder_commit=prov.get("builder_commit"),
                builder_dirty=prov.get("builder_dirty"),
                pipeline_commit=prov.get("pipeline_commit"),
                awpy_version=prov.get("awpy_version"),
                demoparser2_version=prov.get("demoparser2_version"),
                parsed_files=parsed_repo_paths,
                parsed_bytes=parsed_bytes,
            )

        if dry_run:
            rprint(f"  [yellow][DRY] skip upload of {len(artifacts)} files "
                   f"({bytes_total/1e6:.1f} MB) -> {repo_prefix}/ "
                   f"[{schema_version}][/yellow]")
            return _mk_entry(
                [f"{repo_prefix}/{p.name}" for p in artifacts
                 if p.suffix == ".pt"],
                next((f"{repo_prefix}/{p.name}" for p in artifacts
                      if p.name.startswith("feature_schema")), None),
            )

        rprint(f"  [bold]upload[/bold] -> {repo_prefix}/ ({bytes_total/1e6:.1f} MB)")
        try:
            repo_paths = _upload_artifacts(
                api, repo, artifacts, repo_prefix,
                commit_message=(
                    f"Add tick_sequences for {entry.team1} vs {entry.team2} "
                    f"({entry.event}) — {n_rounds} rounds, "
                    f"match {entry.match_id}, {schema_version}"
                ),
            )
        except Exception as e:
            raise StageFailure("upload", str(e)) from e
        return _mk_entry(
            [p for p in repo_paths if p.endswith(".pt")],
            next((p for p in repo_paths if p.rsplit("/", 1)[-1]
                  .startswith("feature_schema")), None),
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
    archive_parsed: bool = True,
    from_parsed: bool = False,
    retry_failed: bool = False,
    allow_dirty: bool = False,
) -> None:
    """Iterate the demo manifest, run tick-sequence build for each unprocessed
    (or stale-schema) match, push results match-by-match with crash-safe
    manifest discipline.
    """
    # --- preflight: builder + parser env guards (before any HF traffic) ----
    schema_version = _builder_schema_version(chimera_dir)
    awpy_v, demoparser_v = _chimera_env_versions(chimera_dir)
    builder_commit = _git_head(chimera_dir)
    builder_dirty = _git_dirty(chimera_dir, [
        "scripts/build_tick_sequences.py", "scripts/parse_demos.py"])
    pipeline_commit = _git_head(Path(__file__).resolve().parent.parent)
    if builder_dirty and not allow_dirty:
        raise SystemExit(
            f"chimera builder scripts have UNCOMMITTED edits "
            f"(commit {builder_commit}+dirty) — bakes would be "
            f"unreproducible. Commit them or pass --allow-dirty.")
    prov = {
        "schema_version": schema_version,
        "builder_commit": builder_commit,
        "builder_dirty": builder_dirty,
        "pipeline_commit": pipeline_commit,
        "awpy_version": awpy_v,
        "demoparser2_version": demoparser_v,
    }
    rprint(f"[bold]Builder:[/bold] {schema_version} "
           f"(chimera {builder_commit or '?'}"
           f"{'+dirty' if builder_dirty else ''}, "
           f"pipeline {pipeline_commit or '?'}, "
           f"awpy {awpy_v}, demoparser2 {demoparser_v})")

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

    fail_mf = Manifest(api, repo, path=FAILURES_MANIFEST_PATH,
                       entry_cls=FailureEntry)
    fail_mf.load()
    if fail_mf.entries:
        rprint(f"  {len(fail_mf.entries)} matches with recorded failures")

    # Pending = never baked OR baked under a different schema version.
    # The latter is the datasheet-mandated corpus-rebuild mechanism: when
    # the builder's SCHEMA_VERSION bumps (e.g. the site fix), every old
    # match becomes pending again WITHOUT deleting the manifest.
    pending: list[ManifestEntry] = []
    stale = 0
    for e in demo_mf.entries:
        old = ts_mf.get(e.match_id)
        if old is None:
            pending.append(e)
        elif old.schema_version != schema_version:
            pending.append(e)
            stale += 1
    rprint(f"[bold]Pending: {len(pending)} matches[/bold] "
           f"({stale} stale-schema re-bakes) (cap this run: {max_matches})")

    processed_this_run = 0
    skipped_failed = 0
    failed: list[tuple[int, str]] = []
    for entry in pending:
        if processed_this_run >= max_matches:
            break
        if should_skip_failed(fail_mf, entry.match_id, retry_failed):
            skipped_failed += 1
            continue
        rprint(f"\n[cyan]→ match {entry.match_id}: {entry.team1} vs {entry.team2} "
               f"({entry.event})[/cyan]")
        t0 = time.time()
        try:
            prev = ts_mf.get(entry.match_id)
            new_entry = process_one(
                api, repo, entry, chimera_dir, downsample, dry_run,
                prov=prov, archive_parsed=archive_parsed,
                from_parsed=from_parsed, prev_entry=prev,
            )
            if new_entry is None:
                failed.append((entry.match_id, "no_output"))
                if not dry_run:
                    record_failure(fail_mf, entry.match_id, "parse", "no_output")
                    fail_mf.push(commit_message=f"failures: match {entry.match_id}")
                continue
            if not dry_run:
                if prev is not None:      # re-bake: replace the stale line
                    ts_mf.entries.remove(prev)
                ts_mf.add(new_entry)
                ts_mf.push(commit_message=(
                    f"tick_sequences manifest: +match {entry.match_id} "
                    f"({schema_version})"
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
            stage = getattr(e, "stage", "bake")
            rprint(f"  [red]failed [{stage}]: {type(e).__name__}: {e}[/red]")
            failed.append((entry.match_id, f"{type(e).__name__}: {e}"))
            if not dry_run:
                try:
                    record_failure(fail_mf, entry.match_id, stage,
                                   f"{type(e).__name__}: {e}"[:500])
                    fail_mf.push(commit_message=f"failures: match {entry.match_id}")
                except Exception as pe:
                    rprint(f"  [red]could not persist failure: {pe}[/red]")

    rprint(f"\n[bold green]Done.[/bold green] "
           f"processed={processed_this_run} failed={len(failed)} "
           f"skipped_failed={skipped_failed} "
           f"pending_remaining={max(0, len(pending) - processed_this_run)}")
    if failed:
        rprint("[red]Failures:[/red]")
        for mid, why in failed:
            rprint(f"  {mid}: {why}")
