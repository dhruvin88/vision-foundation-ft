"""Shared utilities for Modal training scripts."""

from __future__ import annotations

from pathlib import Path


def modal_ignore(path: Path) -> bool:
    """Return True for paths that should be excluded from Modal image mounts."""
    parts    = set(path.parts)
    path_str = path.as_posix()
    return bool(parts & {".venv", "__pycache__", ".git", ".pytest_cache", "checkpoints"}) or \
           any(s in path_str for s in ["experiments/datasets", "experiments/results",
                                        "experiments/results_modal"])
