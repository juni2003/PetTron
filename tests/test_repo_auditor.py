"""
tests/test_repo_auditor.py — Failure reporting for Part A
==========================================================
When the LLM dies, the audit must not be reported as "no issues found": that
is what made a fully broken run look like a clean day in pettron-logs.
"""

import unittest
from unittest import mock

from src.repo_auditor import RepoAuditor
from src.schemas import CodeAnalysisResult


def auditor() -> RepoAuditor:
    return RepoAuditor(gh=mock.Mock(), llm=mock.Mock(), repos_config={}, run_state={})


class TestAuditFailureReporting(unittest.TestCase):
    def test_failed_audit_is_recorded_and_not_reported_as_clean(self):
        aud = auditor()
        cfg = {"name": "o/r"}
        aud._select_repos = lambda: [cfg]  # type: ignore[method-assign]
        aud._audit_repo = mock.Mock(  # type: ignore[method-assign]
            side_effect=RuntimeError("Both Gemini and Groq failed")
        )

        results = aud.run()

        self.assertEqual(
            aud.failed_repos, {"o/r": "Both Gemini and Groq failed"}
        )
        self.assertEqual([name for name, _, _ in results], ["o/r"])

    def test_failed_repo_advances_the_rotation(self):
        aud = auditor()
        cfg = {"name": "o/r"}
        aud._select_repos = lambda: [cfg]  # type: ignore[method-assign]
        aud._audit_repo = mock.Mock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        aud.run()

        # last_checked is what main.py writes back to config/repos.yaml; a repo
        # that cannot be audited must not keep the slot for ever.
        self.assertIn("last_checked", cfg)

    def test_one_failing_repo_does_not_stop_the_others(self):
        aud = auditor()
        bad, good = {"name": "o/bad"}, {"name": "o/good"}
        aud._select_repos = lambda: [bad, good]  # type: ignore[method-assign]
        aud._audit_repo = mock.Mock(  # type: ignore[method-assign]
            side_effect=[
                RuntimeError("LLM down"),
                CodeAnalysisResult(
                    issue_found=True,
                    category="bug",
                    severity="low",
                    target_file="app.py",
                    pr_title="fix: drop unused import",
                    reasoning="clear",
                ),
            ]
        )

        results = aud.run()

        self.assertEqual(list(aud.failed_repos), ["o/bad"])
        by_name = {name: analysis for name, _, analysis in results}
        self.assertTrue(by_name["o/good"].issue_found)
        self.assertFalse(by_name["o/bad"].issue_found)
