
import subprocess
from pathlib import Path
from typing import Optional


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


def validate_revision(repo_path: Path, revision: str) -> str:
    """Verify that a Git revision exists and return its resolved 40-character commit hash.

    Args:
        repo_path: Resolved Path to the Git repository root.
        revision:  Any Git revision specifier (commit hash, branch, tag, etc.).

    Returns:
        The full 40-character commit hash of the resolved revision.

    Raises:
        GitServiceError: If the revision string is empty, the revision does not exist
                         in the repository, or the underlying Git command fails.
    """
    if not revision or not revision.strip():
        raise GitServiceError("Revision must not be empty.")

    try:
        output = _run_git(
            repo_path,
            "rev-parse",
            "--verify",
            f"{revision.strip()}^{{commit}}",
        )
        return output.strip()
    except GitServiceError as exc:
        raise GitServiceError(f"Invalid Git revision: '{revision}'") from exc


def get_file_content_at_revision(
    repository_path: str,
    revision: str,
    file_path: str,
) -> Optional[str]:
    """Retrieve the raw text content of a file at a specific Git revision.

    Uses a two-step approach:

      1. ``git ls-tree -z`` — verifies that the path exists **and** is a regular
         file (blob). Output is NUL-delimited to prevent unquoting or escaping
         issues with spaces, quotes, and special characters. Empty stdout means
         absent; a non-blob entry (tree, submodule) raises :exc:`GitServiceError`.
      2. ``git show <commit>:<path>`` — retrieves the exact file content.

    This separates four distinct situations that must never be conflated:

    - **File does not exist**: returns ``None``.
    - **File exists but is empty** (0-byte): returns ``""`` (empty string).
    - **Path is a directory / submodule**: raises :exc:`GitServiceError`.
    - **Git error** (bad repo, bad revision, execution failure): raises
      :exc:`GitServiceError`.

    Paths containing spaces or special characters are handled safely: all
    arguments are passed as a list, ``-z`` prevents Git from escaping pathnames,
    and the TAB separator isolates the filename from metadata.
    ``shell=False`` is enforced by the underlying ``_run_git`` helper.

    Args:
        repository_path: Absolute path to the Git repository root.
        revision:        Any Git revision specifier (commit hash, branch, tag, etc.).
        file_path:       Repository-relative POSIX path to the file, e.g.
                         ``"src/calculator.py"`` or ``"docs/my file.md"``.

    Returns:
        str:  Raw text of the file at that revision.
        None: When the file does not exist in the revision's tree.

    Raises:
        GitServiceError: If the repository path is invalid, the revision does not
                         exist, the file path argument is empty, the path refers
                         to a directory or non-blob object, or any Git command
                         fails unexpectedly.
    """
    if not file_path or not file_path.strip():
        raise GitServiceError("File path must not be empty.")

    repo_path = validate_repository(repository_path)

    # Validate revision first so an invalid revision raises GitServiceError
    # rather than being confused with a missing-file result.
    resolved_hash = validate_revision(repo_path, revision)

    # Normalise path separators: Git always expects POSIX-style forward slashes
    # even on Windows.
    norm_path = file_path.strip().replace("\\", "/")

    # Step 1: check whether a blob exists in the revision tree for this exact path.
    #
    # Using `-z` outputs NUL-delimited records:
    #   <mode> SP <type> SP <object-hash> TAB <name> \0
    #
    # Advantages of `-z`:
    #   1. Pathnames are output verbatim without octal escaping or C-style quoting.
    #   2. Records are terminated by \0, handling special characters safely.
    #   3. TAB separates metadata from the filename; filenames with spaces are parsed safely.
    #
    # `git ls-tree` exits 0 regardless of whether the path exists:
    #   - empty stdout  -> path does not exist in the tree  -> return None
    #   - non-empty     -> path matched; inspect parsed records
    # An invalid revision would have been caught by validate_revision() already.
    tree_entry = _run_git(repo_path, "ls-tree", "-z", resolved_hash, "--", norm_path)
    if not tree_entry:
        return None

    # Split records on NUL byte
    records = [record for record in tree_entry.split("\0") if record]
    if not records:
        return None

    # If the output returned entries inside norm_path (e.g. norm_path/subfile),
    # then norm_path is a directory (tree), not a file.
    for record in records:
        if "\t" not in record:
            raise GitServiceError(
                f"Unexpected git ls-tree output for '{file_path}': {record!r}"
            )

        meta, entry_name = record.split("\t", 1)
        if entry_name.startswith(norm_path + "/"):
            raise GitServiceError(
                f"Path '{file_path}' refers to a directory (tree), not a file."
            )

        if entry_name == norm_path:
            meta_parts = meta.split()
            if len(meta_parts) < 3:
                raise GitServiceError(
                    f"Unexpected git ls-tree output for '{file_path}': {record!r}"
                )

            entry_type = meta_parts[1]
            if entry_type == "tree":
                raise GitServiceError(
                    f"Path '{file_path}' refers to a directory (tree), not a file."
                )
            if entry_type != "blob":
                raise GitServiceError(
                    f"Path '{file_path}' refers to a {entry_type}, not a file."
                )
            break
    else:
        raise GitServiceError(
            f"git ls-tree returned unexpected entries for '{file_path}'."
        )

    # Step 2: retrieve the exact content.
    # Using the resolved full hash avoids any ambiguity with branch names.
    # An existing 0-byte file produces an empty string with exit code 0.
    return _run_git(repo_path, "show", f"{resolved_hash}:{norm_path}")
