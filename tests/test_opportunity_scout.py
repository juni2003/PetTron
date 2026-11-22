"""
tests/test_opportunity_scout.py — Unit tests for src/opportunity_scout.py
==========================================================================
No network: the GitHub session and GraphQL requester are faked.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.opportunity_scout import DISCUSSION_SEARCH_QUERY, OpportunityScout
from src.schemas import OpportunityDraft, OpportunityReport

PROFILE = {
    "personal": {
        "name": "Junaid Mohi-Ud-Din",
        "availability": "Open to part-time freelance",
        "pricing": {"hourly_rate_usd": "15-30"},
    },
    "identity": "AI & Automation Engineer",
    "core_stack": {"languages": ["Python"]},
    "flagship_projects": [{"name": "NexusAI", "tech": ["FastAPI"], "highlights": ["95% acc"]}],
    "outreach_style": {},
}


class FakeResponse:
    def __init__(self, items, status_code=200):
        self._items = items
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {"items": self._items}


def make_scout(requester=None) -> OpportunityScout:
    profile_path = Path(tempfile.mkdtemp(prefix="pettron_scout_")) / "profile.json"
    profile_path.write_text(json.dumps(PROFILE), encoding="utf-8")

    gh = mock.MagicMock()
    gh.requester = requester or mock.MagicMock()
    scout = OpportunityScout.__new__(OpportunityScout)
    scout.gh = gh
    scout.llm = mock.MagicMock()
    scout.profile = scout._load_profile(profile_path)
    scout._session = mock.MagicMock()
    return scout


def issue(**overrides):
    base = {
        "html_url": "https://github.com/o/r/issues/1",
        "title": "Need an agentic AI dev",
        "repository_url": "https://api.github.com/repos/o/r",
        "body": "Paid bounty, budget 500 USD",
        "comments": 2,
        "created_at": "2026-11-14T00:00:00Z",
        "labels": [{"name": "help wanted"}],
        "state": "open",
    }
    base.update(overrides)
    return base


class TestFilters(unittest.TestCase):
    def test_open_issue_passes(self):
        self.assertTrue(OpportunityScout._passes_filter(issue()))

    def test_closed_locked_pr_and_crowded_are_rejected(self):
        self.assertFalse(OpportunityScout._passes_filter(issue(state="closed")))
        self.assertFalse(OpportunityScout._passes_filter(issue(locked=True)))
        self.assertFalse(
            OpportunityScout._passes_filter(issue(pull_request={"url": "x"}))
        )
        self.assertFalse(OpportunityScout._passes_filter(issue(comments=99)))

    def test_extract_fields_truncates_and_is_missing_key_safe(self):
        out = OpportunityScout._extract_fields(issue(body="x" * 5000))
        self.assertEqual(len(out["body"]), 800)
        self.assertEqual(out["repo"], "o/r")
        self.assertEqual(out["labels"], ["help wanted"])

        sparse = OpportunityScout._extract_fields({"labels": [{"no_name": 1}, "junk"]})
        self.assertEqual(sparse["labels"], [""])
        self.assertEqual(sparse["comments"], 0)


class TestSearch(unittest.TestCase):
    def test_results_are_deduplicated_and_capped(self):
        scout = make_scout()
        same = issue()
        scout._session.get.return_value = FakeResponse([same, issue(html_url="https://x/2")])
        with mock.patch("src.opportunity_scout.time.sleep"):
            results = scout._search_github()
        self.assertEqual(len(results), 2)

    def test_query_failure_is_contained(self):
        scout = make_scout()
        scout._session.get.side_effect = RuntimeError("boom")
        with mock.patch("src.opportunity_scout.time.sleep"):
            self.assertEqual(scout._search_github(), [])

    def test_rate_limit_is_retried_once(self):
        scout = make_scout()
        scout._session.get.side_effect = [
            FakeResponse([], status_code=403),
            FakeResponse([issue()]),
        ]
        with mock.patch("src.opportunity_scout.time.sleep"):
            items = scout._run_search_query("q")
        self.assertEqual(len(items), 1)


class TestDiscussions(unittest.TestCase):
    def test_discussion_nodes_are_normalised(self):
        requester = mock.MagicMock()
        # The shape PyGithub really returns: the whole GraphQL envelope.
        requester.graphql_query.return_value = (
            {},
            {
                "data": {
                    "search": {
                        "nodes": [
                            {
                                "url": "https://github.com/o/r/discussions/7",
                                "title": "Hiring an AI automation engineer",
                                "body": "b" * 2000,
                                "createdAt": "2026-11-13T00:00:00Z",
                                "comments": {"totalCount": 3},
                                "repository": {"nameWithOwner": "o/r"},
                                "labels": {"nodes": [{"name": "hiring"}]},
                            },
                            {"url": None},
                        ]
                    }
                }
            },
        )
        scout = make_scout(requester=requester)
        out = scout._search_discussions("2026-11-08")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["repo"], "o/r")
        self.assertEqual(out[0]["comments"], 3)
        self.assertEqual(len(out[0]["body"]), 800)
        self.assertEqual(requester.graphql_query.call_args[0][0], DISCUSSION_SEARCH_QUERY)

    def test_unwrapped_payload_is_also_accepted(self):
        requester = mock.MagicMock()
        requester.graphql_query.return_value = (
            {},
            {"search": {"nodes": [{"url": "https://github.com/o/r/discussions/9"}]}},
        )
        scout = make_scout(requester=requester)
        out = scout._search_discussions("2026-11-08")
        self.assertEqual([i["url"] for i in out], ["https://github.com/o/r/discussions/9"])

    def test_discussion_failure_is_ignored(self):
        requester = mock.MagicMock()
        requester.graphql_query.side_effect = RuntimeError("no scope")
        scout = make_scout(requester=requester)
        self.assertEqual(scout._search_discussions("2026-11-08"), [])

    def test_empty_payload_is_safe(self):
        requester = mock.MagicMock()
        requester.graphql_query.return_value = (None, None)
        scout = make_scout(requester=requester)
        self.assertEqual(scout._search_discussions("2026-11-08"), [])


class TestRanking(unittest.TestCase):
    def test_prompt_contains_candidates_and_profile_rate(self):
        scout = make_scout()
        scout.llm.call.return_value = OpportunityReport(opportunities=[])

        candidates = [OpportunityScout._extract_fields(issue())]
        scout._rank_and_draft(candidates)

        prompt = scout.llm.call.call_args[0][0]
        self.assertIn("Need an agentic AI dev", prompt)
        self.assertIn("15-30", prompt)
        self.assertIn("never invent a rate", prompt)


if __name__ == "__main__":
    unittest.main()


class TestRankingPrompt(unittest.TestCase):
    def test_discussions_reach_the_ranking_prompt(self):
        """Regression: a fixed [:5] slice cut every discussion out of the prompt."""
        scout = make_scout()
        scout.llm.call.return_value = OpportunityReport(opportunities=[])

        candidates = [issue() for _ in range(10)]
        candidates.append({
            "kind": "discussion",
            "url": "https://github.com/o/r/discussions/7",
            "title": "Freelance project inquiry",
            "repo": "o/r",
            "body": "Looking for someone to build an AI agent.",
            "labels": [],
        })

        scout._rank_and_draft(candidates)

        prompt = scout.llm.call.call_args[0][0]
        self.assertIn("https://github.com/o/r/discussions/7", prompt)
        self.assertIn('"kind": "discussion"', prompt)
        self.assertIn("MUST be one of the top 3", prompt)

    def test_prompt_shows_issues_as_issues(self):
        scout = make_scout()
        scout.llm.call.return_value = OpportunityReport(opportunities=[])
        scout._rank_and_draft([issue()])
        prompt = scout.llm.call.call_args[0][0]
        self.assertIn('"kind": "issue"', prompt)


class TestEmptyRankingIsRetried(unittest.TestCase):
    """gpt-oss answered 'Do not pad' with an empty list on a full candidate pool."""

    def _scout(self, responses):
        scout = make_scout()
        scout.llm.call.side_effect = responses
        return scout

    def test_empty_report_is_asked_once_more(self):
        empty = OpportunityReport(opportunities=[])
        filled = OpportunityReport(
            opportunities=[
                OpportunityDraft(
                    issue_url="https://github.com/o/r/issues/1",
                    issue_title="Need an agent dev",
                    repo_name="o/r",
                    match_score=8,
                    match_reason="agentic work",
                    reply_draft="Happy to help.",
                )
            ]
        )
        scout = self._scout([empty, filled])

        report = scout._rank_and_draft([issue()])

        self.assertEqual(len(report.opportunities), 1)
        self.assertEqual(scout.llm.call.call_count, 2)
        self.assertIn("returned an empty list", scout.llm.call.call_args[0][0])

    def test_a_real_empty_pool_is_not_retried(self):
        scout = self._scout([OpportunityReport(opportunities=[])])

        report = scout._rank_and_draft([])

        self.assertEqual(report.opportunities, [])
        self.assertEqual(scout.llm.call.call_count, 1)
