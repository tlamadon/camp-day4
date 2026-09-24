"""Provenance stamped into every output file.

Each parquet records the SHA-256 of SPEC.md and the git commit it was built
from, so `make check` can tell whether an output still matches the current spec
and code without rerunning the simulation.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

SPEC_FILENAME = "SPEC.md"


def repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor of `start` holding SPEC.md, falling back to the package root."""
    candidates = [Path.cwd() if start is None else Path(start), Path(__file__).resolve()]
    for candidate in candidates:
        for directory in [candidate, *candidate.resolve().parents]:
            if (directory / SPEC_FILENAME).is_file():
                return directory
    return Path(__file__).resolve().parents[2]


def spec_path(start: Path | None = None) -> Path:
    return repo_root(start) / SPEC_FILENAME


def spec_sha256(path: Path | None = None) -> str:
    target = Path(path) if path is not None else spec_path()
    if not target.is_file():
        return "missing"
    return hashlib.sha256(target.read_bytes()).hexdigest()


def git_commit(root: Path | None = None) -> str:
    """HEAD's short-ish sha, suffixed '-dirty' when tracked files have changes."""
    cwd = Path(root) if root is not None else repo_root()

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    sha = git("rev-parse", "HEAD")
    if sha is None:
        return "unknown"
    # Untracked files (output/, .venv/) must not count as a modified tree.
    status = git("status", "--porcelain", "--untracked-files=no")
    return f"{sha}-dirty" if status else sha


def provenance(root: Path | None = None) -> dict[str, str]:
    base = repo_root(root)
    return {
        "spec_sha256": spec_sha256(base / SPEC_FILENAME),
        "git_commit": git_commit(base),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
