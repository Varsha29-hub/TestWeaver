"""Unit tests for the Git unified diff analyzer service."""

import unittest

from services.diff_analyzer import (
    DiffHunk,
    FileChangeType,
    FileDiff,
    ParsedDiff,
    parse_unified_diff,
)


class TestDiffAnalyzer(unittest.TestCase):
    """Comprehensive test suite for unified diff parsing."""

    def test_empty_diff(self):
        """Verify empty or whitespace-only diffs return an empty ParsedDiff."""
        self.assertEqual(len(parse_unified_diff("").files), 0)
        self.assertEqual(len(parse_unified_diff("   \n\n  ").files), 0)

    def test_single_file_modification(self):
        """Verify parsing a standard single file modification diff."""
        diff_text = """diff --git a/src/calculator.py b/src/calculator.py
index 1234567..89abcdef 100644
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -3,4 +3,5 @@ def add(a, b):
 context line
-old line
+new line 1
+new line 2
 trailing context
"""
        parsed = parse_unified_diff(diff_text)
        self.assertEqual(len(parsed.files), 1)

        file_diff = parsed.files[0]
        self.assertEqual(file_diff.old_path, "src/calculator.py")
        self.assertEqual(file_diff.new_path, "src/calculator.py")
        self.assertEqual(file_diff.change_type, FileChangeType.MODIFIED)
        self.assertEqual(len(file_diff.hunks), 1)

        hunk = file_diff.hunks[0]
        self.assertEqual(hunk.old_start, 3)
        self.assertEqual(hunk.old_count, 4)
        self.assertEqual(hunk.new_start, 3)
        self.assertEqual(hunk.new_count, 5)
        self.assertEqual(hunk.heading, "def add(a, b):")
        self.assertEqual(hunk.old_range, (3, 6))
        self.assertEqual(hunk.new_range, (3, 7))

        # Deleted line in old: 3 (context) + 1 = 4
        self.assertEqual(hunk.deleted_lines, [4])
        # Added lines in new: 3 (context) + 1 = 4, 5
        self.assertEqual(hunk.added_lines, [4, 5])
        self.assertFalse(hunk.is_pure_addition)
        self.assertFalse(hunk.is_pure_deletion)

    def test_hunk_header_omitted_counts(self):
        """Verify hunks where line counts are omitted default to 1."""
        # 1. Both counts omitted: @@ -1 +1 @@
        diff_1 = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
"""
        parsed_1 = parse_unified_diff(diff_1)
        hunk_1 = parsed_1.files[0].hunks[0]
        self.assertEqual(hunk_1.old_start, 1)
        self.assertEqual(hunk_1.old_count, 1)
        self.assertEqual(hunk_1.new_start, 1)
        self.assertEqual(hunk_1.new_count, 1)
        self.assertEqual(hunk_1.deleted_lines, [1])
        self.assertEqual(hunk_1.added_lines, [1])

        # 2. Old count omitted: @@ -15 +18,3 @@
        diff_2 = """diff --git a/b.txt b/b.txt
--- a/b.txt
+++ b/b.txt
@@ -15 +18,3 @@
-replaced
+line 1
+line 2
+line 3
"""
        parsed_2 = parse_unified_diff(diff_2)
        hunk_2 = parsed_2.files[0].hunks[0]
        self.assertEqual(hunk_2.old_count, 1)
        self.assertEqual(hunk_2.new_count, 3)
        self.assertEqual(hunk_2.deleted_lines, [15])
        self.assertEqual(hunk_2.added_lines, [18, 19, 20])

        # 3. New count omitted: @@ -20,2 +25 @@
        diff_3 = """diff --git a/c.txt b/c.txt
--- a/c.txt
+++ b/c.txt
@@ -20,2 +25 @@
-del 1
-del 2
+single added
"""
        parsed_3 = parse_unified_diff(diff_3)
        hunk_3 = parsed_3.files[0].hunks[0]
        self.assertEqual(hunk_3.old_count, 2)
        self.assertEqual(hunk_3.new_count, 1)
        self.assertEqual(hunk_3.deleted_lines, [20, 21])
        self.assertEqual(hunk_3.added_lines, [25])

    def test_zero_length_ranges(self):
        """Verify handling of zero-length ranges (pure additions and pure deletions)."""
        # Pure insertion between lines: @@ -10,0 +11,2 @@
        diff_text = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -10,0 +11,2 @@
+inserted 1
+inserted 2
"""
        parsed = parse_unified_diff(diff_text)
        hunk = parsed.files[0].hunks[0]
        self.assertEqual(hunk.old_count, 0)
        self.assertEqual(hunk.new_count, 2)
        self.assertEqual(hunk.old_range, (10, 10))
        self.assertEqual(hunk.new_range, (11, 12))
        self.assertEqual(hunk.deleted_lines, [])
        self.assertEqual(hunk.added_lines, [11, 12])
        self.assertTrue(hunk.is_pure_addition)
        self.assertFalse(hunk.is_pure_deletion)

        # Pure deletion: @@ -30,3 +31,0 @@
        diff_del = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -30,3 +31,0 @@
-removed 1
-removed 2
-removed 3
"""
        parsed_del = parse_unified_diff(diff_del)
        hunk_del = parsed_del.files[0].hunks[0]
        self.assertEqual(hunk_del.old_count, 3)
        self.assertEqual(hunk_del.new_count, 0)
        self.assertEqual(hunk_del.new_range, (31, 31))
        self.assertEqual(hunk_del.deleted_lines, [30, 31, 32])
        self.assertEqual(hunk_del.added_lines, [])
        self.assertTrue(hunk_del.is_pure_deletion)
        self.assertFalse(hunk_del.is_pure_addition)

    def test_added_file(self):
        """Verify diff of a newly created file."""
        diff_text = """diff --git a/src/new_mod.py b/src/new_mod.py
new file mode 100644
index 0000000..abcdef1
--- /dev/null
+++ b/src/new_mod.py
@@ -0,0 +1,3 @@
+def func():
+    return 42
+
"""
        parsed = parse_unified_diff(diff_text)
        self.assertEqual(len(parsed.files), 1)
        file_diff = parsed.files[0]
        self.assertIsNone(file_diff.old_path)
        self.assertEqual(file_diff.new_path, "src/new_mod.py")
        self.assertEqual(file_diff.path, "src/new_mod.py")
        self.assertEqual(file_diff.change_type, FileChangeType.ADDED)
        self.assertEqual(file_diff.all_added_lines, [1, 2, 3])
        self.assertEqual(file_diff.all_deleted_lines, [])

    def test_deleted_file(self):
        """Verify diff of a deleted file."""
        diff_text = """diff --git a/src/legacy.py b/src/legacy.py
deleted file mode 100644
index abcdef1..0000000
--- a/src/legacy.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def deprecated():
-    pass
"""
        parsed = parse_unified_diff(diff_text)
        self.assertEqual(len(parsed.files), 1)
        file_diff = parsed.files[0]
        self.assertEqual(file_diff.old_path, "src/legacy.py")
        self.assertIsNone(file_diff.new_path)
        self.assertEqual(file_diff.path, "src/legacy.py")
        self.assertEqual(file_diff.change_type, FileChangeType.DELETED)
        self.assertEqual(file_diff.all_deleted_lines, [1, 2])
        self.assertEqual(file_diff.all_added_lines, [])

    def test_renamed_file(self):
        """Verify diff of a renamed file with modifications."""
        diff_text = """diff --git a/old_name.py b/new_name.py
similarity index 90%
rename from old_name.py
rename to new_name.py
index 1111111..2222222 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,2 +1,2 @@
-name = 'old'
+name = 'new'
"""
        parsed = parse_unified_diff(diff_text)
        file_diff = parsed.files[0]
        self.assertEqual(file_diff.old_path, "old_name.py")
        self.assertEqual(file_diff.new_path, "new_name.py")
        self.assertEqual(file_diff.change_type, FileChangeType.RENAMED)
        self.assertEqual(file_diff.all_deleted_lines, [1])
        self.assertEqual(file_diff.all_added_lines, [1])

    def test_multiple_hunks_in_one_file(self):
        """Verify multiple hunks in a single file maintain accurate line numbering."""
        diff_text = """diff --git a/service.py b/service.py
index 1234567..89abcdef 100644
--- a/service.py
+++ b/service.py
@@ -10,3 +10,4 @@ def first_method():
 ctx
-old line 11
+new line 11
+extra line 12
 ctx
@@ -35,3 +36,3 @@ def second_method():
 ctx
-old line 36
+new line 37
 ctx
"""
        parsed = parse_unified_diff(diff_text)
        file_diff = parsed.files[0]
        self.assertEqual(len(file_diff.hunks), 2)

        # Hunk 1
        self.assertEqual(file_diff.hunks[0].deleted_lines, [11])
        self.assertEqual(file_diff.hunks[0].added_lines, [11, 12])

        # Hunk 2
        self.assertEqual(file_diff.hunks[1].deleted_lines, [36])
        self.assertEqual(file_diff.hunks[1].added_lines, [37])

        # Aggregated properties
        self.assertEqual(file_diff.all_deleted_lines, [11, 36])
        self.assertEqual(file_diff.all_added_lines, [11, 12, 37])

    def test_multiple_files_in_diff(self):
        """Verify diff containing multiple files parses into distinct FileDiff records."""
        diff_text = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-a
+A
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-b
+B
"""
        parsed = parse_unified_diff(diff_text)
        self.assertEqual(len(parsed.files), 2)
        self.assertEqual(parsed.changed_paths, ["a.py", "b.py"])

        file_a = parsed.get_file("a.py")
        self.assertIsNotNone(file_a)
        self.assertEqual(file_a.all_added_lines, [1])

        file_b = parsed.get_file("b.py")
        self.assertIsNotNone(file_b)
        self.assertEqual(file_b.all_added_lines, [1])

    def test_no_newline_at_eof_marker(self):
        """Verify '\\ No newline at end of file' does not corrupt line numbering."""
        diff_text = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 line 1
-line 2
\\ No newline at end of file
+line 2
+line 3
\\ No newline at end of file
"""
        parsed = parse_unified_diff(diff_text)
        hunk = parsed.files[0].hunks[0]
        self.assertEqual(hunk.deleted_lines, [2])
        self.assertEqual(hunk.added_lines, [2, 3])

    def test_real_sample_repo_diff(self):
        """Verify parsing against the exact diff from sample_repo commits 9c27d50 -> 2bffdde."""
        sample_diff = """diff --git a/src/calculator.py b/src/calculator.py
index 015b4a2..3508e4e 100644
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -3,4 +3,8 @@ def add(a, b):
 
 
 def subtract(a, b):
-    return a - b
\\ No newline at end of file
+    return a - b
+
+
+def multiply(a, b):
+    return a * b
\\ No newline at end of file
"""
        parsed = parse_unified_diff(sample_diff)
        self.assertEqual(len(parsed.files), 1)

        f = parsed.files[0]
        self.assertEqual(f.path, "src/calculator.py")
        self.assertEqual(f.change_type, FileChangeType.MODIFIED)

        hunk = f.hunks[0]
        self.assertEqual(hunk.old_start, 3)
        self.assertEqual(hunk.old_count, 4)
        self.assertEqual(hunk.new_start, 3)
        self.assertEqual(hunk.new_count, 8)
        self.assertEqual(hunk.deleted_lines, [6])
        self.assertEqual(hunk.added_lines, [6, 7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
