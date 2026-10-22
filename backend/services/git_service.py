
import subprocess
from pathlib import Path


class GitServiceError(Exception):
    """Raised when a Git repository cannot be analyzed."""


def _run_git(repo_path: Path, *args: str) -> str:
    """Run Git safely and return its standard output."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitServiceError("Git command timed out.") from exc
    except OSError as exc:
        raise GitServiceError("Unable to execute Git.") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or "Git command failed."
        raise GitServiceError(message)

    return result.stdout


def validate_repository(repository_path: str) -> Path:
    """Return the resolved path if it is a Git working tree."""
    path = Path(repository_path).expanduser()

    if not path.exists():
        raise GitServiceError("Repository path does not exist.")

    if not path.is_dir():
        raise GitServiceError("Repository path must be a directory.")

    try:
        resolved_path = path.resolve(strict=True)
        result = _run_git(
            resolved_path,
            "rev-parse",
            "--is-inside-work-tree",
        )
    except (OSError, RuntimeError) as exc:
        raise GitServiceError("Unable to access repository path.") from exc

    if result.strip() != "true":
        raise GitServiceError("Path is not a Git working tree.")

    return resolved_path


def get_changed_files(
    repository_path: str,
    base_revision: str,
    target_revision: str,
) -> list[str]:
    """Return filenames changed between two Git revisions."""
    repo_path = validate_repository(repository_path)

    if not base_revision.strip() or not target_revision.strip():
        raise GitServiceError("Both revisions must be provided.")

    output = _run_git(
        repo_path,
        "diff",
        "--name-only",
        "--no-renames",
        base_revision,
        target_revision,
        "--",
    )

    return [line for line in output.splitlines() if line]


def get_diff(
    repository_path: str,
    base_revision: str,
    target_revision: str,
) -> str:
    """Return the textual diff between two Git revisions."""
    repo_path = validate_repository(repository_path)

    if not base_revision.strip() or not target_revision.strip():
        raise GitServiceError("Both revisions must be provided.")

    return _run_git(
        repo_path,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        base_revision,
        target_revision,
        "--",
    )
