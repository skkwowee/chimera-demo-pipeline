"""HF dataset upload of .dem files (with hf_transfer for parallel upload)."""

from __future__ import annotations

import os
from pathlib import Path

# Enable parallel chunked uploads — significant speedup on large files
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, CommitOperationAdd


def upload_demos(api: HfApi, repo_id: str, dem_paths: list[Path],
                  repo_type: str = "dataset",
                  commit_message: str | None = None,
                  match_id: int | None = None) -> list[str]:
    """Upload a list of local .dem files to the HF repo in a SINGLE commit
    (atomic per-match — either all maps of a series land or none do).

    With `match_id`, files land at demos/<match_id>/<name>.dem — canonical
    filenames contain no match id, so a rematch of the same teams on the
    same map slot would otherwise silently OVERWRITE the earlier match's
    .dem at the flat demos/<name>.dem path (create_commit replaces existing
    paths without warning). Old flat paths keep working for reads via the
    manifest's recorded demo_files.

    Returns the list of `path_in_repo` strings on success.
    """
    if not dem_paths:
        return []
    prefix = f"demos/{match_id}" if match_id is not None else "demos"
    ops = []
    repo_paths = []
    for p in dem_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        repo_path = f"{prefix}/{p.name}"
        repo_paths.append(repo_path)
        ops.append(CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(p),
        ))
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=ops,
        commit_message=commit_message
        or f"Add {len(dem_paths)} demo files ({sum(p.stat().st_size for p in dem_paths)/1e6:.0f} MB)",
    )
    return repo_paths
