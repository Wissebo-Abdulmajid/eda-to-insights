from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 of a file in a streaming way (safe for large files).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

