"""Filesystem helpers: output-dir creation, checkpoint paths, CSV naming."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if missing, return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiment_artifact_path(output_dir: str | Path, name: str, suffix: str) -> Path:
    """Build ``{output_dir}/{name}.{suffix}`` after ensuring output_dir exists."""
    return ensure_dir(output_dir) / f"{name}.{suffix}"
