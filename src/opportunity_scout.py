"""
opportunity_scout.py — PetTron Part B: Opportunity Scout
==========================================================
Responsible for:
  1. Querying the GitHub Search API with 5 targeted issue queries plus one
     GraphQL query for Discussions, to find open freelance / help-wanted /
     bounty / agentic-AI posts.
  2. Filtering raw results (remove stale, closed, locked, or spammy threads).
  3. Deduplicating across queries.
  4. Passing the top 10 results + the user's profile.json to the LLM.
  5. Getting back an OpportunityReport with the top 3 ranked + drafted replies.

GitHub Search API limit (authenticated): 30 requests/minute.
With 5 queries + small sleep buffers, we stay well within limits.
"""

import json
import logging
import time
from pathlib import Path

import requests
from github import Github

from src.llm_client import LLMClient
from src.schemas import OpportunityReport

logger = logging.getLogger(__name__)

# ── Search configuration ──────────────────────────────────────────────────────

# Issues created within the last N days
SEARCH_WINDOW_DAYS = 7

# Skip issues with more than this many comments (too crowded, low chance of success)
MAX_COMMENTS = 15

# Max *issue* results to collect before passing to LLM for ranking
MAX_RAW_RESULTS = 10

# Discussions get their own budget on top of the issues, so a full issue pool
# can no longer push every discussion out of the ranking.
MAX_DISCUSSION_RESULTS = 4

# Candidates handed to the ranking LLM (issues + discussions).
MAX_RANKED_CANDIDATES = 14

# Courtesy sleep between GitHub Search API calls (in seconds)
SEARCH_SLEEP_SECS = 3

SEARCH_QUERIES = [
    # Format: q=<search_terms> type:issue state:open created:>YYYY-MM-DD
    'label:"help wanted" (FastAPI OR LangGraph OR RAG OR "agentic AI") language:Python type:issue state:open',
    '"looking for developer" (FastAPI OR Next.js OR Python) type:issue state:open',
    '"bounty" (Python OR FastAPI OR LLM OR RAG) type:issue state:open',
    '"freelance" (React OR Next.js OR FastAPI OR "full stack") type:issue state:open',
    '"hiring" (AI OR LLM OR "machine learning" OR Python) type:issue state:open',
]

# GitHub Discussions are only reachable through GraphQL (no REST search).
DISCUSSION_SEARCH_QUERY = """
query($q: String!) {
  search(query: $q, type: DISCUSSION, first: 10) {
    nodes {
      ... on Discussion {
        title
        url
        createdAt
        body
        comments { totalCount }
        repository { nameWithOwner }
        labels(first: 5) { nodes { name } }
      }
    }
  }
}
"""


class OpportunityScout:
    """
    Runs Part B of PetTron: searching GitHub for freelance opportunities
    and generating tailored reply drafts.

    Usage (from main.py):
        scout = OpportunityScout(gh_client, llm_client, profile_path)
        report: OpportunityReport = scout.run()
    """

    def __init__(self, gh: Github, llm: LLMClient, profile_path: Path) -> None:
        self.gh = gh
        self.llm = llm
        self.profile = self._load_profile(profile_path)
        self._session = self._make_session()

    def run(self) -> OpportunityReport:
        """
        Main entry point.
        Returns OpportunityReport with the top 3 opportunities + reply drafts.
        """
        raw_results = self._search_github()

        if not raw_results:
            logger.warning("[Scout] No opportunities found in search results.")
            return OpportunityReport(opportunities=[])

        logger.info(f"[Scout] Found {len(raw_results)} candidate issues. Sending to LLM for ranking ...")
        report: OpportunityReport = self._rank_and_draft(raw_results)
        logger.info(f"[Scout] LLM selected {len(report.opportunities)} top opportunities.")
        return report

    # ── Private methods ────────────────────────────────────────────────────────

    def _load_profile(self, profile_path: Path) -> dict:
        """Load profile.json for the LLM ranking context."""
        with open(profile_path, encoding="utf-8") as f:
            return json.load(f)

    def _make_session(self) -> requests.Session:
        """Create an authenticated requests.Session for the GitHub Search API."""
        import os
        pat = os.environ.get("GH_PAT", "").strip()
        session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        # Never send "Authorization: token " (an empty token is a 401).
        if pat:
            headers["Authorization"] = f"token {pat}"
        session.headers.update(headers)
        return session

    def _search_github(self) -> list[dict]:
        """
        Run all SEARCH_QUERIES and return a deduplicated list of issue dicts.
        """
        from datetime import date, timedelta

        cutoff = (date.today() - timedelta(days=SEARCH_WINDOW_DAYS)).isoformat()
        seen_urls: set[str] = set()
        all_results: list[dict] = []

        for query in SEARCH_QUERIES:
            if len(all_results) >= MAX_RAW_RESULTS:
                break

            full_query = f"{query} created:>{cutoff}"
            logger.info(f"[Scout] Searching: {full_query[:80]}...")

            try:
                items = self._run_search_query(full_query)
                for item in items:
                    url = item.get("html_url", "")
                    if url not in seen_urls and self._passes_filter(item):
                        seen_urls.add(url)
                        all_results.append(self._extract_fields(item))
                        if len(all_results) >= MAX_RAW_RESULTS:
                            break
            except Exception as exc:
                logger.warning(f"[Scout] Search query failed: {exc}")

            time.sleep(SEARCH_SLEEP_SECS)  # Respect GitHub's rate limits

        logger.info(f"[Scout] Collected {len(all_results)} candidate issue(s) after filtering.")

        # GitHub Discussions are GraphQL-only, so they are searched separately.
        # This is additive: if it fails, issue results are still used.
        discussions = 0
        for item in self._search_discussions(cutoff):
            if discussions >= MAX_DISCUSSION_RESULTS:
                break
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            all_results.append(item)
            discussions += 1

        logger.info(f"[Scout] Final candidate pool: {len(all_results)} item(s).")
        return all_results

    def _search_discussions(self, cutoff: str) -> list[dict]:
        """
        Search GitHub Discussions for AI/automation hiring or help-wanted posts.

        Discussions have no REST endpoint, so this uses GraphQL. Any failure
        (missing scope, API change, network) is logged and ignored.
        """
        query = (
            '("agentic AI" OR "AI automation" OR "help wanted" OR "freelance") '
            f"created:>{cutoff}"
        )
        try:
            _, data = self.gh.requester.graphql_query(
                DISCUSSION_SEARCH_QUERY, {"q": query}
            )
        except Exception as exc:
            logger.warning(f"[Scout] Discussion search skipped: {exc}")
            return []

        # PyGithub hands back the whole GraphQL envelope ("{"data": {...}}"),
        # not the inner "data" object. Accept both so the shape can never make
        # every discussion silently disappear again.
        payload = data if isinstance(data, dict) else {}
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        nodes = (payload.get("search") or {}).get("nodes") or []
        results: list[dict] = []
        for node in nodes:
            if not isinstance(node, dict) or not node.get("url"):
                continue
            results.append({
                "kind":       "discussion",
                "url":        node["url"],
                "title":      node.get("title", ""),
                "repo":       (node.get("repository") or {}).get("nameWithOwner", ""),
                "body":       (node.get("body") or "")[:800],
                "comments":   (node.get("comments") or {}).get("totalCount", 0),
                "created_at": node.get("createdAt", ""),
                "labels":     [
                    l.get("name", "")
                    for l in ((node.get("labels") or {}).get("nodes") or [])
                    if isinstance(l, dict)
                ],
            })

        logger.info(f"[Scout] Discussion search returned {len(results)} candidate(s).")
        return results

    def _run_search_query(self, query: str) -> list[dict]:
        """Execute one GitHub Search API request, retrying once on rate limit."""
        for attempt in (1, 2):
            response = self._session.get(
                "https://api.github.com/search/issues",
                params={"q": query, "sort": "created", "order": "desc", "per_page": 10},
                timeout=15,
            )
            if response.status_code == 403 and attempt == 1:
                logger.warning("[Scout] GitHub Search API rate limit hit. Sleeping 60s ...")
                time.sleep(60)
                continue
            response.raise_for_status()
            return response.json().get("items", [])
        return []

    @staticmethod
    def _passes_filter(item: dict) -> bool:
        """Return True if this issue passes our quality filters."""
        # Skip closed or locked issues
        if item.get("state") != "open":
            return False
        if item.get("locked"):
            return False

        # Skip overpopulated threads (too competitive)
        if item.get("comments", 0) > MAX_COMMENTS:
            return False

        # Skip pull requests (the Search API sometimes returns PRs too)
        if "pull_request" in item:
            return False

        return True

    @staticmethod
    def _extract_fields(item: dict) -> dict:
        """Extract only the fields we need from a raw GitHub API item."""
        return {
            "url":        item.get("html_url", ""),
            "title":      item.get("title", ""),
            "repo":       item.get("repository_url", "").replace(
                "https://api.github.com/repos/", ""
            ),
            "body":       (item.get("body") or "")[:800],   # Truncate long descriptions
            "comments":   item.get("comments", 0),
            "created_at": item.get("created_at", ""),
            "labels":     [l.get("name", "") for l in item.get("labels", []) if isinstance(l, dict)],
        }

    def _rank_and_draft(self, candidates: list[dict]) -> OpportunityReport:
        """
        Send the top candidates + profile to the LLM.
        Ask it to rank and return the top 3 with drafted replies.
        """
        # Truncate bodies to keep the prompt tight. Discussions used to be cut
        # off here by a fixed [:5] slice — they are appended after the issues,
        # so a full issue pool pushed every discussion out of the prompt.
        condensed_candidates = [
            {
                "kind": c.get("kind", "issue"),
                "url": c.get("url", ""),
                "title": c.get("title", ""),
                "repo": c.get("repo", ""),
                "snippet": (c.get("body") or "")[:250],
                "labels": c.get("labels", []),
            }
            for c in candidates[:MAX_RANKED_CANDIDATES]
        ]

        # Compact profile context
        condensed_profile = {
            "name": self.profile.get("personal", {}).get("name", "Junaid Mohi-Ud-Din"),
            "availability": self.profile.get("personal", {}).get("availability", ""),
            "pricing": self.profile.get("personal", {}).get("pricing", {}),
            "identity": self.profile.get("identity", ""),
            "core_skills": self.profile.get("core_stack", {}),
            "flagship_projects": [
                {
                    "name": p.get("name"),
                    "tech": p.get("tech"),
                    "highlights": p.get("highlights", [])[:2],
                }
                for p in self.profile.get("flagship_projects", [])[:4]
            ],
            "outreach_rules": self.profile.get("outreach_style", {}),
        }

        candidates_text = json.dumps(condensed_candidates, indent=2)
        profile_text = json.dumps(condensed_profile, indent=2)

        prompt = f"""You are PetTron, an AI career assistant for a developer named Junaid Mohi-Ud-Din.

Your task: From the candidate GitHub issues below, select the TOP 3 that best match Junaid's skills
and would be most likely to result in a paid gig or collaboration. Then write a tailored,
human-sounding reply draft for each one.

JUNAID'S PROFILE:
{profile_text}

CANDIDATE GITHUB ISSUES:
{candidates_text}

RULES FOR SELECTING OPPORTUNITIES:
  - Prioritise posts that explicitly mention compensation (bounty, paid, freelance, hiring).
  - Rank by: relevance to Junaid's stack > posting recency > likelihood of response.
  - Skip posts that look like generic bug reports with no intention to hire.
  - Candidates have a "kind": "issue" or "discussion". If a relevant discussion is in the
    list, it MUST be one of the top 3 — discussions are where people ask for AI-agent,
    automation and freelance help and where a reply is most likely to be read.
  - Do not pad: if fewer than 3 candidates are genuinely relevant, return fewer.

RULES FOR REPLY DRAFTS:
  - Each reply must be CONCISE (2-3 short paragraphs max).
  - Start by acknowledging their SPECIFIC problem (not generic "I saw your post").
  - Mention exactly ONE of Junaid's projects that is directly relevant, with one real metric.
  - State availability and pricing using ONLY the values in Junaid's profile above — never invent a rate.
  - End with a clear, low-pressure call to action ("happy to share my portfolio / schedule a call").
  - Do NOT start with "I am a developer looking for work." — that's spam.
  - Sound like a confident, helpful engineer, not a job seeker.
"""
        return self.llm.call(prompt, OpportunityReport)

