"""
tests/test_verifier.py — Safety & patching tests for src/verifier.py
=====================================================================
These cover the parts that must never regress: forbidden-path blocking,
secret redaction, and exact-match patching. No network or git is needed.
"""

import tempfile
import unittest
from pathlib import Path

from src.schemas import CodeAnalysisResult
from src.verifier import PatchError, VerificationError, Verifier


def make_verifier() -> Verifier:
    # gh/llm are unused by the methods under test.
    return Verifier(gh=None, llm=None)


class TestForbiddenPaths(unittest.TestCase):
    def setUp(self):
        self.v = make_verifier()

    def test_workflow_files_are_refused(self):
        for path in (
            ".github/workflows/ci.yml",
            ".github/workflows/daily_agent.yml",
            "./.github/workflows/x.yaml",
        ):
            with self.assertRaises(VerificationError, msg=path):
                self.v._check_forbidden(path)

    def test_secret_files_are_refused(self):
        for path in (".env", "config/.env", "app/.env.local", "secrets/key.txt", "secrets"):
            with self.assertRaises(VerificationError, msg=path):
                self.v._check_forbidden(path)

    def test_windows_separators_and_case_are_normalized(self):
        with self.assertRaises(VerificationError):
            self.v._check_forbidden(".github\\workflows\\ci.yml")
        with self.assertRaises(VerificationError):
            self.v._check_forbidden("Secrets/Key.txt")

    def test_dot_prefix_does_not_bypass(self):
        for path in ("./.github/workflows/x.yaml", "src/../.github/workflows/ci.yml"):
            with self.assertRaises(VerificationError, msg=path):
                self.v._check_forbidden(path)

    def test_paths_escaping_the_repo_are_refused(self):
        for path in ("../outside.py", "src/../../etc/passwd", "..", ""):
            with self.assertRaises(VerificationError, msg=path):
                self.v._check_forbidden(path)

    def test_normal_source_files_are_allowed(self):
        for path in ("src/app.py", "tests/test_app.py", "env.example", "docs/.github/workflows.md"):
            self.v._check_forbidden(path)  # must not raise


class TestRedact(unittest.TestCase):
    def test_pat_in_url_is_scrubbed(self):
        text = "fatal: https://ghp_abcdefghijklmnop1234@github.com/o/r.git not found"
        out = Verifier._redact(text)
        self.assertNotIn("ghp_abcdefghijklmnop1234", out)
        self.assertIn("//***@", out)

    def test_bare_tokens_are_scrubbed(self):
        for token in ("ghp_ABCDEFGHIJ1234", "github_pat_11ABCDEFG0_rest"):
            self.assertNotIn(token, Verifier._redact(f"token {token} rejected"))

    def test_empty_input(self):
        self.assertEqual(Verifier._redact(""), "")
        self.assertEqual(Verifier._redact(None), "")


class TestApplyPatch(unittest.TestCase):
    def test_exact_match_is_replaced_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "app.py"
            f.write_text("a = 1\nb = 2\nb = 2\n", encoding="utf-8")

            new = Verifier._apply_patch(f, "b = 2", "b = 3")

            self.assertEqual(new, "a = 1\nb = 3\nb = 2\n")
            self.assertEqual(f.read_text(encoding="utf-8"), new)

    def test_missing_block_raises_and_leaves_file_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "app.py"
            original = "a = 1\n"
            f.write_text(original, encoding="utf-8")

            with self.assertRaises(PatchError):
                Verifier._apply_patch(f, "not here", "x")

            self.assertEqual(f.read_text(encoding="utf-8"), original)


class TestBranchName(unittest.TestCase):
    def test_slug_is_safe_and_prefixed(self):
        analysis = CodeAnalysisResult(
            issue_found=True, pr_title="Fix: SQL Injection in User Lookup!!"
        )
        name = Verifier._make_branch_name(analysis)
        self.assertTrue(name.startswith("bot/fix-"))
        self.assertNotIn(" ", name)
        self.assertNotIn(":", name)
        self.assertNotIn("!", name)

    def test_falls_back_to_category_when_no_title(self):
        analysis = CodeAnalysisResult(issue_found=True, category="security")
        self.assertTrue(Verifier._make_branch_name(analysis).endswith("-security"))


if __name__ == "__main__":
    unittest.main()
