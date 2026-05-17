"""HF dataset upload of .dem files (with hf_transfer for parallel upload)."""

from __future__ import annotations

import os
from pathlib import Path

# Enable parallel chunked uploads — significant speedup on large files
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, CommitOperationAdd


def upload_demos(api: HfApi, repo_id: str, dem_paths: list[Path],
                  repo_type: str = "dataset",
                  commit_message: str | None = None) -> list[str]:
    """Upload a list of local .dem files to demos/<name>.dem on the HF repo
    in a SINGLE commit (atomic per-match — either all maps of a series land
    or none do).

    Returns the list of `path_in_repo` strings on success.
    """
    if not dem_paths:
        return []
    ops = []
    repo_paths = []
    for p in dem_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        repo_path = f"demos/{p.name}"
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
