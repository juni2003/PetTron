"""
tests/test_verifier.py — Safety & patching tests for src/verifier.py
=====================================================================
These cover the parts that must never regress: forbidden-path blocking,
secret redaction, and exact-match patching. No network or git is needed.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class TestPatchMismatchSelfCorrection(unittest.TestCase):
    """
    Regression: a hallucinated search_block used to abort the PR outright with
    "search_block not found verbatim". The LLM now gets the real file and one
    chance to redo the patch.
    """

    FILE = "def total(xs):\n    return sum(xs)\n"

    def _repo(self, tmp: str) -> Path:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        (repo / "app.py").write_text(self.FILE, encoding="utf-8")
        return repo

    def _analysis(self, search: str, replace: str) -> CodeAnalysisResult:
        return CodeAnalysisResult(
            issue_found=True,
            category="bug",
            severity="low",
            target_file="app.py",
            search_block=search,
            replace_block=replace,
            pr_title="fix: guard empty input",
            reasoning="test",
        )

    def test_wrong_block_is_retried_with_the_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(tmp)
            llm = mock.MagicMock()
            llm.call.return_value = self._analysis(self.FILE, "def total(xs):\n    return sum(xs or [])\n")
            verifier = Verifier(gh=None, llm=llm)

            content, branch, used = verifier._attempt_verify(
                repo,
                "o/r",
                {"verify_cmd": "python -m compileall . -q"},
                self._analysis("def absent():\n    return None\n", "def total(xs):\n    return 0\n"),
                attempt=1,
            )

            self.assertIn("sum(xs or [])", content)
            self.assertTrue(branch.startswith("bot/fix-"))
            self.assertEqual(used.search_block, self.FILE)
            prompt = llm.call.call_args[0][0]
            self.assertIn(self.FILE.strip(), prompt)
            self.assertIn("NOT found in the file", prompt)

    def test_still_wrong_after_correction_gives_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(tmp)
            llm = mock.MagicMock()
            llm.call.return_value = self._analysis("still not in the file", "x = 1\n")
            verifier = Verifier(gh=None, llm=llm)

            with self.assertRaises(VerificationError):
                verifier._attempt_verify(
                    repo,
                    "o/r",
                    {"verify_cmd": "python -m compileall . -q"},
                    self._analysis("not in the file", "x = 1\n"),
                    attempt=1,
                )
            self.assertEqual(len(llm.call.call_args_list), 1)

    def test_llm_declining_the_fix_gives_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(tmp)
            llm = mock.MagicMock()
            llm.call.return_value = CodeAnalysisResult(issue_found=False, reasoning="cannot")
            verifier = Verifier(gh=None, llm=llm)

            with self.assertRaises(VerificationError):
                verifier._attempt_verify(
                    repo,
                    "o/r",
                    {"verify_cmd": "python -m compileall . -q"},
                    self._analysis("not in the file", "x = 1\n"),
                    attempt=1,
                )


class TestBaselineVerify(unittest.TestCase):
    """An already-broken verify_cmd must be reported as such, not as a bad patch."""

    def test_failing_baseline_raises_with_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(gh=None, llm=None)
            with self.assertRaises(VerificationError) as ctx:
                verifier._assert_baseline_verify(
                    Path(tmp), {"verify_cmd": "exit 2"}
                )
            self.assertIn("unmodified repo", str(ctx.exception))
            self.assertIn("exit 2", str(ctx.exception))

    def test_passing_baseline_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(gh=None, llm=None)
            verifier._assert_baseline_verify(Path(tmp), {"verify_cmd": "true"})

    def test_baseline_can_be_disabled_per_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(gh=None, llm=None)
            verifier._assert_baseline_verify(
                Path(tmp), {"verify_cmd": "exit 9", "baseline_verify": False}
            )

    def test_verify_output_is_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            returncode, output = Verifier._run_verify(
                Path(tmp), "echo 'token ghp_abcdefghijklmnopqrstuvwxyz' && exit 1"
            )
            self.assertEqual(returncode, 1)
            self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz", output)


class TestLowValuePatchGate(unittest.TestCase):
    """Opening a PR for a cosmetic patch is worse than opening none."""

    def _apply(self, content, search, replace, name="mod.py"):
        path = Path(tempfile.mkdtemp(prefix="pettron_gate_")) / name
        path.write_text(content, encoding="utf-8")
        try:
            Verifier._apply_patch(path, search, replace)
        except PatchError as exc:
            return f"refused: {exc}"
        return "applied"

    def test_comment_only_patch_is_refused(self):
        out = self._apply("def f():\n    pass\n", "    pass\n", "    # TODO\n")
        self.assertIn("refused", out)

    def test_docstring_only_patch_is_refused(self):
        content = 'def f():\n    """Doc."""\n    return 1\n'
        out = self._apply(content, '    """Doc."""\n', '    """Better docs."""\n')
        self.assertIn("refused", out)

    def test_javascript_literal_in_python_is_refused(self):
        out = self._apply("x = None\n", "x = None\n", "x = false\n")
        self.assertIn("JavaScript literal", out)

    def test_real_behaviour_change_is_applied(self):
        out = self._apply("def f():\n    return 1\n", "    return 1\n", "    return 2\n")
        self.assertEqual(out, "applied")

    def test_try_except_wrapping_is_applied(self):
        out = self._apply(
            "a = 1\n", "a = 1\n", "try:\n    a = 1\nexcept Exception:\n    a = 0\n"
        )
        self.assertEqual(out, "applied")
