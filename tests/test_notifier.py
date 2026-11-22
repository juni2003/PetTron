"""
tests/test_notifier.py — Unit tests for src/notifier.py
========================================================
The digest is the only thing the user actually reads each morning, so the
formatting and the "one issue per day" behaviour are covered here.
"""

import unittest
from unittest import mock

from src.notifier import Notifier
from src.schemas import OpportunityDraft, OpportunityReport, PRRecord


def draft(i: int = 1) -> OpportunityDraft:
    return OpportunityDraft(
        issue_url=f"https://github.com/o/r/issues/{i}",
        issue_title=f"Hiring AI engineer #{i}",
        repo_name="o/r",
        match_score=8,
        match_reason="Direct match on agentic AI + FastAPI.",
        reply_draft="Hey — I've shipped a LangGraph agent that cut latency 40%. Happy to help.",
    )


def pr() -> PRRecord:
    return PRRecord(
        repo="o/r",
        pr_url="https://github.com/o/r/pull/9",
        pr_title="fix: handle empty response",
        category="bug",
        severity="medium",
        opened_at="2026-11-15T00:00:00+00:00",
    )


class TestBuildBody(unittest.TestCase):
    def test_all_sections_present(self):
        body = Notifier._build_body(
            today="2026-11-15",
            prs_opened=[pr()],
            repos_skipped=["o/skipped"],
            repos_no_issue=["o/clean"],
            opportunity_report=OpportunityReport(opportunities=[draft(1), draft(2)]),
            run_errors=["Opportunity Scout failed: boom"],
            duration_secs=95.0,
            llm_used="gemini:gemini-2.0-flash",
        )

        self.assertIn("Part A", body)
        self.assertIn("https://github.com/o/r/pull/9", body)
        self.assertIn("o/skipped", body)
        self.assertIn("o/clean", body)
        self.assertIn("Hiring AI engineer #1", body)
        self.assertIn("Hiring AI engineer #2", body)
        self.assertIn("LangGraph agent", body)  # the copy-pasteable draft
        self.assertIn("Opportunity Scout failed: boom", body)
        self.assertIn("1m 35s", body)
        self.assertIn("gemini:gemini-2.0-flash", body)

    def test_empty_report_still_produces_readable_body(self):
        body = Notifier._build_body(
            today="2026-11-15",
            prs_opened=[],
            repos_skipped=[],
            repos_no_issue=[],
            opportunity_report=OpportunityReport(opportunities=[]),
            run_errors=[],
            duration_secs=5.0,
            llm_used="not-called",
        )
        self.assertIn("No PRs opened today.", body)
        self.assertIn("No strong opportunities found", body)
        self.assertIn("0m 5s", body)


class TestSendDigest(unittest.TestCase):
    def _notifier(self, existing_issues):
        gh = mock.MagicMock()
        repo = gh.get_repo.return_value
        repo.get_issues.return_value = existing_issues
        return Notifier(gh, "owner/pettron-logs"), repo

    def test_creates_one_issue_when_none_exists_today(self):
        notifier, repo = self._notifier([])
        url = notifier.send_digest(
            prs_opened=[pr()],
            repos_skipped=[],
            repos_no_issue=[],
            opportunity_report=OpportunityReport(opportunities=[draft()]),
            run_errors=[],
            duration_secs=1.0,
            llm_used="gemini:gemini-2.0-flash",
        )
        repo.create_issue.assert_called_once()
        self.assertEqual(url, repo.create_issue.return_value.html_url)

    def test_second_run_same_day_appends_comment_instead_of_new_issue(self):
        existing = mock.MagicMock()
        existing.pull_request = None
        existing.title = "🤖 PetTron Daily Digest — 2026-11-15"
        existing.html_url = "https://github.com/owner/pettron-logs/issues/1"

        notifier, repo = self._notifier([existing])
        with mock.patch("src.notifier.datetime") as dt:
            dt.now.return_value.strftime.return_value = "2026-11-15"
            url = notifier.send_digest(
                prs_opened=[],
                repos_skipped=[],
                repos_no_issue=[],
                opportunity_report=OpportunityReport(opportunities=[]),
                run_errors=[],
                duration_secs=1.0,
                llm_used="not-called",
            )

        existing.create_comment.assert_called_once()
        repo.create_issue.assert_not_called()
        self.assertEqual(url, existing.html_url)

    def test_pull_requests_are_not_mistaken_for_the_digest_issue(self):
        pr_like = mock.MagicMock()
        pr_like.pull_request = object()  # a PR, not an issue
        pr_like.title = "🤖 PetTron Daily Digest — 2026-11-15"

        notifier, repo = self._notifier([pr_like])
        with mock.patch("src.notifier.datetime") as dt:
            dt.now.return_value.strftime.return_value = "2026-11-15"
            notifier.send_digest(
                prs_opened=[],
                repos_skipped=[],
                repos_no_issue=[],
                opportunity_report=OpportunityReport(opportunities=[]),
                run_errors=[],
                duration_secs=1.0,
                llm_used="not-called",
            )

        repo.create_issue.assert_called_once()


class TestFailedAudits(unittest.TestCase):
    """A dead LLM must never read as a clean day."""

    def _body(self, repos_no_issue=None, **kwargs) -> str:
        return Notifier._build_body(
            today="2026-11-15",
            prs_opened=[],
            repos_skipped=[],
            repos_no_issue=repos_no_issue or [],
            opportunity_report=OpportunityReport(opportunities=[]),
            run_errors=[],
            duration_secs=2.0,
            llm_used="none — every provider failed",
            **kwargs,
        )

    def test_unaudited_repo_is_not_listed_as_clean(self):
        body = self._body(
            repos_no_issue=["o/clean"],
            repos_failed={"o/broken": "Both Gemini and Groq failed after all retries."},
        )

        self.assertIn("could NOT be audited", body)
        self.assertIn("`o/broken` — Both Gemini and Groq failed", body)
        clean_line = next(line for line in body.splitlines() if "no issues found" in line)
        self.assertIn("o/clean", clean_line)
        self.assertNotIn("o/broken", clean_line)

    def test_failure_reason_is_a_single_line(self):
        body = self._body(
            repos_failed={"o/broken": "line one\nline two\nline three"},
        )

        self.assertIn("`o/broken` — line one", body)
        self.assertNotIn("line two", body.split("## 💼 Part B")[0])

    def test_no_failed_repos_means_no_failure_section(self):
        self.assertNotIn("could NOT be audited", self._body(repos_no_issue=["o/clean"]))


if __name__ == "__main__":
    unittest.main()
