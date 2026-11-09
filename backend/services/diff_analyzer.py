"""Git unified diff analyzer service for TestWeaver.

Phase 1: Accurate parsing of unified diff headers, hunks, line ranges,
and change types (additions, deletions, modifications, renames).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Regex for unified diff hunk header:
# Matches @@ -old_start[,old_count] +new_start[,new_count] @@ [heading]
HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: ?(.*))?$"
)

# Regex for git diff file header:
# Handles both unquoted and quoted file paths
GIT_DIFF_HEADER_PATTERN = re.compile(
    r'^diff --git (?:a/(.*?)|"(?:a/)?(.*?)") (?:b/(.*?)|"(?:b/)?(.*?)")?$'
)


class FileChangeType(str, Enum):
    """Categorization of file-level Git changes."""

    MODIFIED = "MODIFIED"
    ADDED = "ADDED"
    DELETED = "DELETED"
    RENAMED = "RENAMED"


@dataclass
class DiffHunk:
    """Represents a single parsed diff hunk with fine-grained line mapping."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    heading: str = ""
    deleted_lines: list[int] = field(default_factory=list)  # Line numbers in base revision
    added_lines: list[int] = field(default_factory=list)    # Line numbers in target revision
    context_lines: list[str] = field(default_factory=list)

    @property
    def old_range(self) -> tuple[int, int]:
        """Return (start, end) 1-based line interval for base revision."""
        if self.old_count == 0:
            return (self.old_start, self.old_start)
        return (self.old_start, self.old_start + self.old_count - 1)

    @property
    def new_range(self) -> tuple[int, int]:
        """Return (start, end) 1-based line interval for target revision."""
        if self.new_count == 0:
            return (self.new_start, self.new_start)
        return (self.new_start, self.new_start + self.new_count - 1)

    @property
    def is_pure_addition(self) -> bool:
        """True if the hunk only inserts lines without deleting any."""
        return len(self.deleted_lines) == 0 and len(self.added_lines) > 0

    @property
    def is_pure_deletion(self) -> bool:
        """True if the hunk only removes lines without adding any."""
        return len(self.added_lines) == 0 and len(self.deleted_lines) > 0


@dataclass
class FileDiff:
    """Represents changes parsed for a single file in the unified diff."""

    old_path: Optional[str]
    new_path: Optional[str]
    change_type: FileChangeType = FileChangeType.MODIFIED
    hunks: list[DiffHunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        """Return the active file path (new_path if present, else old_path)."""
        return self.new_path or self.old_path or ""

    @property
    def all_added_lines(self) -> list[int]:
        """All line numbers added/modified in target revision across all hunks."""
        lines: list[int] = []
        for hunk in self.hunks:
            lines.extend(hunk.added_lines)
        return sorted(set(lines))

    @property
    def all_deleted_lines(self) -> list[int]:
        """All line numbers deleted in base revision across all hunks."""
        lines: list[int] = []
        for hunk in self.hunks:
            lines.extend(hunk.deleted_lines)
        return sorted(set(lines))


@dataclass
class ParsedDiff:
    """Represents the complete parsed unified diff across all files."""

    files: list[FileDiff] = field(default_factory=list)

    def get_file(self, path: str) -> Optional[FileDiff]:
        """Look up FileDiff by file path (matches either new_path or old_path)."""
        normalized = Path(path).as_posix()
        for f in self.files:
            if f.new_path and Path(f.new_path).as_posix() == normalized:
                return f
            if f.old_path and Path(f.old_path).as_posix() == normalized:
                return f
        return None

    @property
    def changed_paths(self) -> list[str]:
        """Return list of all affected file paths."""
        return [f.path for f in self.files if f.path]


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse a unified git diff string into structured FileDiff and DiffHunk objects.

    Handles:
      - Additions, deletions, modifications, and renames
      - Omitted hunk counts (defaulting to 1 per specification)
      - Zero-length line ranges (e.g. @@ -0,0 +1,5 @@ or @@ -10,0 +11,2 @@)
      - Multiple hunks per file
      - Special EOF markers ('\\ No newline at end of file')
    """
    if not diff_text or not diff_text.strip():
        return ParsedDiff()

    files: list[FileDiff] = []
    current_file: Optional[FileDiff] = None
    current_hunk: Optional[DiffHunk] = None
    cur_old_line = 0
    cur_new_line = 0

    lines = diff_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # 1. New file diff header: diff --git a/... b/...
        if line.startswith("diff --git "):
            current_hunk = None
            old_p = None
            new_p = None

            git_match = GIT_DIFF_HEADER_PATTERN.match(line)
            if git_match:
                old_p = git_match.group(1) or git_match.group(2)
                new_p = git_match.group(3) or git_match.group(4)
            else:
                parts = line.split(" ")
                if len(parts) >= 4:
                    old_p = parts[2].removeprefix("a/")
                    new_p = parts[3].removeprefix("b/")

            current_file = FileDiff(
                old_path=old_p,
                new_path=new_p,
                change_type=FileChangeType.MODIFIED,
                hunks=[],
            )
            files.append(current_file)
            i += 1
            continue

        # 2. File metadata lines (before the first hunk)
        if current_file is not None and current_hunk is None:
            if line.startswith("new file mode"):
                current_file.change_type = FileChangeType.ADDED
                current_file.old_path = None
                i += 1
                continue
            elif line.startswith("deleted file mode"):
                current_file.change_type = FileChangeType.DELETED
                current_file.new_path = None
                i += 1
                continue
            elif line.startswith("rename from "):
                current_file.change_type = FileChangeType.RENAMED
                current_file.old_path = line[len("rename from "):].strip().strip('"')
                i += 1
                continue
            elif line.startswith("rename to "):
                current_file.change_type = FileChangeType.RENAMED
                current_file.new_path = line[len("rename to "):].strip().strip('"')
                i += 1
                continue
            elif line.startswith("--- "):
                path_part = line[4:].strip().strip('"')
                if path_part == "/dev/null":
                    current_file.old_path = None
                    current_file.change_type = FileChangeType.ADDED
                elif path_part.startswith("a/"):
                    current_file.old_path = path_part[2:]
                else:
                    current_file.old_path = path_part
                i += 1
                continue
            elif line.startswith("+++ "):
                path_part = line[4:].strip().strip('"')
                if path_part == "/dev/null":
                    current_file.new_path = None
                    current_file.change_type = FileChangeType.DELETED
                elif path_part.startswith("b/"):
                    current_file.new_path = path_part[2:]
                else:
                    current_file.new_path = path_part
                i += 1
                continue

        # Handle non-git unified diff headers: "--- a/file" without "diff --git"
        if line.startswith("--- ") and current_file is None:
            path_part = line[4:].strip().strip('"')
            old_p = None if path_part == "/dev/null" else (path_part[2:] if path_part.startswith("a/") else path_part)
            current_file = FileDiff(
                old_path=old_p,
                new_path=None,
                change_type=FileChangeType.MODIFIED,
                hunks=[],
            )
            files.append(current_file)
            i += 1
            continue

        if line.startswith("+++ ") and current_file is not None and current_file.new_path is None:
            path_part = line[4:].strip().strip('"')
            new_p = None if path_part == "/dev/null" else (path_part[2:] if path_part.startswith("b/") else path_part)
            current_file.new_path = new_p
            if current_file.old_path is None:
                current_file.change_type = FileChangeType.ADDED
            elif current_file.new_path is None:
                current_file.change_type = FileChangeType.DELETED
            i += 1
            continue

        # 3. Hunk header: @@ -old_start,old_count +new_start,new_count @@
        hunk_match = HUNK_HEADER_PATTERN.match(line)
        if hunk_match:
            if current_file is None:
                current_file = FileDiff(
                    old_path=None,
                    new_path=None,
                    change_type=FileChangeType.MODIFIED,
                    hunks=[],
                )
                files.append(current_file)

            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1
            heading = hunk_match.group(5) or ""

            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                heading=heading,
                deleted_lines=[],
                added_lines=[],
                context_lines=[],
            )
            current_file.hunks.append(current_hunk)

            cur_old_line = old_start
            cur_new_line = new_start
            i += 1
            continue

        # 4. Lines inside a hunk
        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk.added_lines.append(cur_new_line)
                cur_new_line += 1
            elif line.startswith("-"):
                current_hunk.deleted_lines.append(cur_old_line)
                cur_old_line += 1
            elif line.startswith(" "):
                current_hunk.context_lines.append(line[1:])
                cur_old_line += 1
                cur_new_line += 1
            elif line.startswith("\\"):
                # e.g. "\ No newline at end of file" -> do not advance line numbers
                pass
            else:
                # Other unrecognized line marks end of current hunk
                current_hunk = None

        i += 1

    return ParsedDiff(files=files)
