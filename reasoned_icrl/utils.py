"""Package-wide artifact hashing, JSON identities and checkout resolution.

These helpers carry no framework dependency so that every layer -- runtime,
model, teacher and analysis -- can share one implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large artifacts stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    """Serialize to canonical JSON so identities are byte-stable across runs."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def identity(value: object) -> str:
    """Return the content address of any JSON-serializable value."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path: Path, value: object, *, immutable: bool = True) -> str:
    """Atomically write canonical JSON and return its SHA-256.

    With ``immutable`` set, rewriting an existing file with different bytes is
    an error: recorded artifacts are evidence and must not be silently amended.
    """
    data = json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Immutable artifact changed: {path.name}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def git_output(repository: Path, *arguments: str) -> str | None:
    """Run a git command, returning None when git or the repository is absent."""
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_versions(
    names: tuple[str, ...] = (
        "amago",
        "gymnasium",
        "numpy",
        "torch",
        "popgym",
        "flash-attn",
    ),
) -> dict[str, str]:
    """Installed versions of the packages a run's numbers depend on."""
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "unavailable"
    return output


def source_identity(repository: Path) -> dict[str, object]:
    """Hash the checkout precisely enough to tell two runs apart.

    Passive native-run provenance for historical analysis; the DAT sprint does
    not require or validate this identity before execution.
    """
    status = git_output(repository, "status", "--porcelain", "--untracked-files=normal")
    lockfile = repository / "uv.lock"
    return {
        "git_commit": git_output(repository, "rev-parse", "HEAD") or "unavailable",
        "git_dirty": status is None or bool(status),
        "lockfile_sha256": file_sha256(lockfile) if lockfile.is_file() else "",
        "source_sha256": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in sorted((repository / "reasoned_icrl").rglob("*.py"))
        },
    }


def repository_root(start: str | Path | None = None) -> Path:
    """Resolve the checkout root by searching upward for its marker files."""
    here = (Path(start) if start is not None else Path(__file__)).resolve()
    for path in (here, *here.parents):
        if (path / "pyproject.toml").is_file() and (path / "configs").is_dir():
            return path
    raise ValueError(f"Could not resolve the repository root from {here}.")


__all__ = [
    "file_sha256",
    "git_output",
    "identity",
    "json_bytes",
    "package_versions",
    "repository_root",
    "source_identity",
    "write_json",
]
