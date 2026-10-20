from __future__ import annotations

from pathlib import Path

from baselines.pidetector.core import MODEL_ID, MODEL_REVISION


def resolve_snapshot(*, download: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=not download)
    ).resolve()
