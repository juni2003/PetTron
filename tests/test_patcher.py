"""
tests/test_patcher.py — Unit Tests for PetTron's Safe Patch Logic
==================================================================
Tests the core search/replace patching function in verifier.py to
ensure it never corrupts files or silently applies wrong changes.

Runs with:
    python -m unittest discover -s tests -p "*.py"
or:
    pytest tests/ -v
"""

import re
import tempfile
import unittest
from pathlib import Path

from src.schemas import CodeAnalysisResult
from src.verifier import PatchError, Verifier


def make_temp_file(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


class TestApplyPatch(unittest.TestCase):
    """Tests for Verifier._apply_patch() — the core safe search/replace."""

    def test_simple_patch_applied_correctly(self):
        """A basic patch is applied and the file content is updated."""
        original = "def fetch(url):\n    response = requests.get(url)\n    return response.json()\n"
        search = "    response = requests.get(url)\n    return response.json()"
        replace = (
            "    try:\n"
            "        response = requests.get(url, timeout=10)\n"
            "        response.raise_for_status()\n"
            "        return response.json()\n"
            "    except requests.RequestException as e:\n"
            "        raise RuntimeError(f'Request failed: {e}') from e"
        )

        file_path = make_temp_file(original)
        try:
            result = Verifier._apply_patch(file_path, search, replace)
            self.assertIn("raise_for_status", result)
            self.assertIn("timeout=10", result)
            self.assertIn("RuntimeError", result)
            self.assertNotIn("requests.get(url)\n    return response.json()", result)
        finally:
            file_path.unlink(missing_ok=True)

    def test_patch_writes_to_disk(self):
        """After patching, the file on disk reflects the new content."""
        original = "x = 1\ny = 2\n"
        file_path = make_temp_file(original)
        try:
            Verifier._apply_patch(file_path, "x = 1", "x = 42")
            on_disk = file_path.read_text(encoding="utf-8")
            self.assertIn("x = 42", on_disk)
            self.assertNotIn("x = 1", on_disk)
        finally:
            file_path.unlink(missing_ok=True)

    def test_patch_returns_new_content(self):
        """_apply_patch returns the new file content as a string."""
        file_path = make_temp_file("a = 'old'\n")
        try:
            result = Verifier._apply_patch(file_path, "a = 'old'", "a = 'new'")
            self.assertIsInstance(result, str)
            self.assertIn("a = 'new'", result)
        finally:
            file_path.unlink(missing_ok=True)

    def test_only_first_occurrence_replaced(self):
        """When search_block appears twice, only the first occurrence is replaced."""
        original = "x = 1\nx = 1\n"
        file_path = make_temp_file(original)
        try:
            result = Verifier._apply_patch(file_path, "x = 1", "x = 99")
            self.assertEqual(result.count("x = 99"), 1)
            self.assertEqual(result.count("x = 1"), 1)
        finally:
            file_path.unlink(missing_ok=True)

    def test_multiline_patch(self):
        """A multi-line search/replace patch is applied correctly."""
        original = (
            "def calculate(a, b):\n"
            "    result = a / b\n"
            "    return result\n"
        )
        search = "    result = a / b\n    return result"
        replace = (
            "    if b == 0:\n"
            "        raise ValueError('Division by zero')\n"
            "    result = a / b\n"
            "    return result"
        )
        file_path = make_temp_file(original)
        try:
            result = Verifier._apply_patch(file_path, search, replace)
            self.assertIn("Division by zero", result)
            self.assertIn("if b == 0", result)
        finally:
            file_path.unlink(missing_ok=True)

    def test_raises_patch_error_when_not_found(self):
        """PatchError is raised when search_block doesn't exist in the file."""
        file_path = make_temp_file("hello = 'world'\n")
        try:
            with self.assertRaises(PatchError):
                Verifier._apply_patch(file_path, "this_does_not_exist = True", "x = 1")
        finally:
            file_path.unlink(missing_ok=True)

    def test_indentation_only_mismatch_is_applied(self):
        """A unique line-for-line match applies even if indentation drifted."""
        original = "    indented_line = True\n"
        file_path = make_temp_file(original)
        try:
            new_content = Verifier._apply_patch(
                file_path, "      indented_line = True", "    indented_line = False\n"
            )
            self.assertEqual(new_content, "    indented_line = False\n")
        finally:
            file_path.unlink(missing_ok=True)

    def test_ambiguous_whitespace_match_raises(self):
        """Two identical regions mean the match is ambiguous → refuse."""
        original = "x = 1\nif y:\n    x = 1\n"
        file_path = make_temp_file(original)
        try:
            with self.assertRaises(PatchError):
                Verifier._apply_patch(file_path, "\tx = 1", "\tx = 2")
            self.assertEqual(file_path.read_text(), original)
        finally:
            file_path.unlink(missing_ok=True)

    def test_file_unchanged_on_patch_error(self):
        """When PatchError is raised, the original file content is preserved."""
        original = "safe_code = True\n"
        file_path = make_temp_file(original)
        try:
            with self.assertRaises(PatchError):
                Verifier._apply_patch(file_path, "nonexistent_code = True", "broken")
            self.assertEqual(file_path.read_text(encoding="utf-8"), original)
        finally:
            file_path.unlink(missing_ok=True)


class TestBranchNameGeneration(unittest.TestCase):
    """Tests for Verifier._make_branch_name()."""

    def test_branch_starts_with_bot_fix(self):
        analysis = CodeAnalysisResult(
            issue_found=True,
            pr_title="Add error handling in scraper.py",
            category="error_handling",
        )
        branch = Verifier._make_branch_name(analysis)
        self.assertTrue(branch.startswith("bot/fix-"))

    def test_branch_name_is_url_safe(self):
        analysis = CodeAnalysisResult(
            issue_found=True,
            pr_title="Fix: SQL injection in user/login endpoint (critical!)",
            category="security",
        )
        branch = Verifier._make_branch_name(analysis)
        self.assertTrue(re.match(r"^[a-z0-9/\-]+$", branch), f"Unsafe branch name: {branch}")

    def test_branch_name_max_length(self):
        analysis = CodeAnalysisResult(
            issue_found=True,
            pr_title="A" * 200,
            category="bug",
        )
        branch = Verifier._make_branch_name(analysis)
        self.assertLessEqual(len(branch), 60, f"Branch name too long: {branch}")


if __name__ == "__main__":
    unittest.main()
