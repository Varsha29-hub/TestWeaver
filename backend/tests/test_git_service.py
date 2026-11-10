"""Unit tests for the Git service — Stage 1: validate_revision and get_file_content_at_revision.

Every test is completely self-contained, using temporary Git repositories
created during test setup. No hardcoded absolute repository paths or commit
hashes are used.
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from services.git_service import (
    GitServiceError,
    get_file_content_at_revision,
    validate_repository,
    validate_revision,
)


def _cleanup_temp_dir(temp_dir: tempfile.TemporaryDirectory) -> None:
    """Safely remove a temporary directory on Windows, where Git can mark objects read-only."""
    try:
        temp_dir.cleanup()
    except Exception:
        def _remove_readonly(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(temp_dir.name, onerror=_remove_readonly, ignore_errors=True)


class BaseGitTestCase(unittest.TestCase):
    """Base test case providing a self-contained temporary Git repository with known fixtures."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name).resolve()
        self.repo_str = str(self.repo_path)

        # Initialize repository
        subprocess.run(["git", "init", self.repo_str], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", self.repo_str, "config", "user.email", "test@testweaver.internal"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", self.repo_str, "config", "user.name", "TestWeaver Test"],
            check=True,
            capture_output=True,
        )

        # Create test fixture files:
        # 1. Standard python file
        src_dir = self.repo_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        calc_file = src_dir / "calculator.py"
        self.calc_content = (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n"
        )
        calc_file.write_text(self.calc_content, encoding="utf-8")

        # 2. Empty 0-byte file
        keep_dir = self.repo_path / "backend" / "services"
        keep_dir.mkdir(parents=True, exist_ok=True)
        keep_file = keep_dir / ".gitkeep"
        keep_file.write_bytes(b"")

        # 3. File whose path contains spaces
        spaced_file = src_dir / "file with spaces.py"
        self.spaced_content = "def spaced_func():\n    return 'spaces'\n"
        spaced_file.write_text(self.spaced_content, encoding="utf-8")

        # Commit everything
        subprocess.run(["git", "-C", self.repo_str, "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", self.repo_str, "commit", "-m", "Initial test fixtures"],
            check=True,
            capture_output=True,
        )

        # Retrieve commit hash
        rev_out = subprocess.run(
            ["git", "-C", self.repo_str, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.commit_hash = rev_out.stdout.strip()
        self.short_hash = self.commit_hash[:7]

        # Fixture identifiers
        self.file_existing = "src/calculator.py"
        self.file_empty = "backend/services/.gitkeep"
        self.file_with_spaces = "src/file with spaces.py"
        self.dir_existing = "src"
        self.file_missing = "nonexistent_file.py"
        self.invalid_revision = "invalid_rev_xyz_12345"
        self.invalid_repo_str = str(self.repo_path / "nonexistent_repo_dir")

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)
        super().tearDown()


class TestValidateRevision(BaseGitTestCase):
    """Tests for validate_revision(repo_path, revision)."""

    def test_valid_short_hash_returns_full_hash(self):
        """A valid abbreviated commit hash is expanded to 40 hex characters."""
        repo_path = validate_repository(self.repo_str)
        full_hash = validate_revision(repo_path, self.short_hash)
        self.assertEqual(len(full_hash), 40)
        self.assertTrue(full_hash.isalnum(), "Result must be a hex string")
        self.assertEqual(full_hash, self.commit_hash)

    def test_valid_full_hash_round_trips(self):
        """Passing the full 40-char hash returns the same hash."""
        repo_path = validate_repository(self.repo_str)
        full_hash_1 = validate_revision(repo_path, self.short_hash)
        full_hash_2 = validate_revision(repo_path, full_hash_1)
        self.assertEqual(full_hash_1, full_hash_2)

    def test_invalid_revision_raises_git_service_error(self):
        """A revision string that does not exist raises GitServiceError."""
        repo_path = validate_repository(self.repo_str)
        with self.assertRaises(GitServiceError) as ctx:
            validate_revision(repo_path, self.invalid_revision)
        self.assertIn(self.invalid_revision, str(ctx.exception))

    def test_empty_revision_raises_git_service_error(self):
        """An empty revision string raises GitServiceError without calling Git."""
        repo_path = validate_repository(self.repo_str)
        with self.assertRaises(GitServiceError):
            validate_revision(repo_path, "")

    def test_whitespace_only_revision_raises_git_service_error(self):
        """A whitespace-only revision string raises GitServiceError."""
        repo_path = validate_repository(self.repo_str)
        with self.assertRaises(GitServiceError):
            validate_revision(repo_path, "   ")


class TestGetFileContentAtRevision(BaseGitTestCase):
    """Tests for get_file_content_at_revision(repository_path, revision, file_path)."""

    def test_existing_file_returns_string_content(self):
        """Retrieving a known existing file returns its exact source as a string."""
        content = get_file_content_at_revision(
            self.repo_str, self.short_hash, self.file_existing
        )
        self.assertIsInstance(content, str)
        self.assertEqual(content, self.calc_content)
        self.assertIn("def add", content)

    def test_existing_file_content_matches_full_hash(self):
        """Passing the full 40-char hash produces the same content as the short hash."""
        content_short = get_file_content_at_revision(
            self.repo_str, self.short_hash, self.file_existing
        )
        content_full = get_file_content_at_revision(
            self.repo_str, self.commit_hash, self.file_existing
        )
        self.assertEqual(content_short, content_full)

    def test_missing_file_returns_none(self):
        """A file absent from the revision tree returns None, not an error."""
        result = get_file_content_at_revision(
            self.repo_str, self.commit_hash, self.file_missing
        )
        self.assertIsNone(result)

    def test_empty_file_returns_empty_string_not_none(self):
        """A tracked 0-byte file returns empty string; it must NOT return None."""
        content = get_file_content_at_revision(
            self.repo_str, self.commit_hash, self.file_empty
        )
        self.assertIsNotNone(content, "0-byte file must not return None")
        self.assertIsInstance(content, str)
        self.assertEqual(content, "", "0-byte file must return empty string")

    def test_directory_path_raises_git_service_error(self):
        """A path pointing to a directory raises GitServiceError, not returning content or None."""
        with self.assertRaises(GitServiceError) as ctx:
            get_file_content_at_revision(
                self.repo_str, self.commit_hash, self.dir_existing
            )
        self.assertIn("directory", str(ctx.exception).lower())

    def test_path_with_spaces_is_safe(self):
        """Paths containing spaces that do not exist return None safely."""
        result = get_file_content_at_revision(
            self.repo_str, self.commit_hash, "src/missing file with spaces.py"
        )
        self.assertIsNone(result)

    def test_existing_file_with_spaces_returns_content(self):
        """An existing file whose path contains spaces is retrieved correctly with exact content."""
        content = get_file_content_at_revision(
            self.repo_str, self.commit_hash, self.file_with_spaces
        )
        self.assertEqual(content, self.spaced_content)

    def test_invalid_revision_raises_git_service_error(self):
        """An invalid revision raises GitServiceError, not returns None."""
        with self.assertRaises(GitServiceError) as ctx:
            get_file_content_at_revision(
                self.repo_str, self.invalid_revision, self.file_existing
            )
        self.assertIn(self.invalid_revision, str(ctx.exception))

    def test_invalid_repository_raises_git_service_error(self):
        """A non-existent repository path raises GitServiceError."""
        with self.assertRaises(GitServiceError):
            get_file_content_at_revision(
                self.invalid_repo_str, self.commit_hash, self.file_existing
            )

    def test_empty_file_path_raises_git_service_error(self):
        """An empty file_path argument raises GitServiceError."""
        with self.assertRaises(GitServiceError):
            get_file_content_at_revision(self.repo_str, self.commit_hash, "")

    def test_whitespace_only_file_path_raises_git_service_error(self):
        """A whitespace-only file_path raises GitServiceError."""
        with self.assertRaises(GitServiceError):
            get_file_content_at_revision(self.repo_str, self.commit_hash, "   ")


if __name__ == "__main__":
    unittest.main()
