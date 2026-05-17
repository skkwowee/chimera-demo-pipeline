"""Processed-match tracking, stored as JSONL on the HF dataset itself.

Why on HF and not local: the pipeline must be runnable from a fresh RunPod
worker without prior state. Manifest-on-HF makes the *latest commit* the
single source of truth — any worker can resume by downloading the manifest,
filtering, and proceeding.

Layout on HF:
    demos/<team1>-vs-<team2>-m<N>-<map>.dem        # the demo files
    processed_manifest.jsonl                       # one JSON line per match
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi


MANIFEST_PATH = "processed_manifest.jsonl"


@dataclass
class ManifestEntry:
    match_id: int
    slug: str
    team1: str
    team2: str
    event: str
    score: str
    stars: int
    date_unix: int | None
    demo_files: list[str]   # paths in repo, e.g. ["demos/foo-m1-mirage.dem"]
    bytes_total: int
    processed_at_unix: int = field(default_factory=lambda: int(time.time()))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "ManifestEntry":
        d = json.loads(s)
        return cls(**d)


class Manifest:
    """Read-modify-append the JSONL manifest on the HF dataset.

    For efficiency, callers should `load()` once at the start of a batch,
    accumulate entries with `add()`, and `commit_batch()` once at the end
    to push a single combined manifest. Each individual demo upload still
    commits separately (so a kill mid-batch leaves the demos on HF;
    the manifest line is only added when the demo is confirmed uploaded).
    """

    def __init__(self, api: HfApi, repo_id: str, repo_type: str = "dataset"):
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.entries: list[ManifestEntry] = []
        self.processed_ids: set[int] = set()

    def load(self) -> None:
        """Download manifest from HF if it exists. Populates entries + processed_ids."""
        try:
            local = self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=MANIFEST_PATH,
                repo_type=self.repo_type,
            )
            for line in Path(local).read_text().splitlines():
                if line.strip():
                    e = ManifestEntry.from_json(line)
                    self.entries.append(e)
                    self.processed_ids.add(e.match_id)
        except Exception:
            # No manifest yet — fresh start
            pass

    def has(self, match_id: int) -> bool:
        return match_id in self.processed_ids

    def add(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)
        self.processed_ids.add(entry.match_id)

    def push(self, commit_message: str | None = None) -> None:
        """Upload the FULL manifest (all entries) to HF in one commit."""
        body = "\n".join(e.to_json() for e in self.entries) + "\n"
        # upload_file requires a path; write to buffer + use BytesIO via tempfile
        # Actually upload_file accepts path_or_fileobj=bytes too
        self.api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo=MANIFEST_PATH,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=commit_message or f"manifest: {len(self.entries)} matches",
        )
