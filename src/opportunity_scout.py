"""
opportunity_scout.py — PetTron Part B: Opportunity Scout
==========================================================
Responsible for:
  1. Querying the GitHub Search API with 5 targeted search queries
     to find open freelance / help-wanted / bounty issues.
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

# Max results to collect total before passing to LLM for ranking
MAX_RAW_RESULTS = 10

# Courtesy sleep between GitHub Search API calls (in seconds)
SEARCH_SLEEP_SECS = 3

SEARCH_QUERIES = [
    # Format: q=<search_terms> type:issue state:open created:>YYYY-MM-DD
    'label:"help wanted" (FastAPI OR LangGraph OR RAG OR "agentic AI") language:Python',
    '"looking for developer" (FastAPI OR Next.js OR Python) type:issue state:open',
    '"bounty" (Python OR FastAPI OR LLM OR RAG) type:issue state:open',
    '"freelance" (React OR Next.js OR FastAPI OR "full stack") type:issue state:open',
    '"hiring" (AI OR LLM OR "machine learning" OR Python) type:issue state:open',
]


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
        try:
            self._gh_token = self.gh.get_user().login
        except Exception as e:
            logger.warning(f"[Scout] GitHub user check notice: {e}")
            self._gh_token = ""
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
        session.headers.update({
            "Authorization": f"token {pat}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
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
        return all_results

    def _run_search_query(self, query: str) -> list[dict]:
        """Execute one GitHub Search API request and return items list."""
        response = self._session.get(
            "https://api.github.com/search/issues",
            params={"q": query, "sort": "created", "order": "desc", "per_page": 10},
            timeout=15,
        )
        if response.status_code == 403:
            logger.warning("[Scout] GitHub Search API rate limit hit. Sleeping 60s ...")
            time.sleep(60)
            return []
        response.raise_for_status()
        return response.json().get("items", [])

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
            "labels":     [l["name"] for l in item.get("labels", [])],
        }

    def _rank_and_draft(self, candidates: list[dict]) -> OpportunityReport:
        """
        Send the top candidates + profile to the LLM.
        Ask it to rank and return the top 3 with drafted replies.
        """
        # Take top 5 candidates and truncate bodies to keep prompt tight
        condensed_candidates = []
        for c in candidates[:5]:
            condensed_candidates.append({
                "url": c.get("url", ""),
                "title": c.get("title", ""),
                "repo": c.get("repo", ""),
                "snippet": (c.get("body") or "")[:250],
                "labels": c.get("labels", []),
            })

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
  - Prioritise issues that explicitly mention compensation (bounty, paid, freelance, hiring).
  - Rank by: relevance to Junaid's stack > posting recency > likelihood of response.
  - Skip issues that look like generic bug reports with no intention to hire.

RULES FOR REPLY DRAFTS:
  - Each reply must be CONCISE (2-3 short paragraphs max).
  - Start by acknowledging their SPECIFIC problem (not generic "I saw your post").
  - Mention exactly ONE of Junaid's projects that is directly relevant, with one real metric.
  - State availability (part-time / open to full-time) and pricing (flexible, $15-30/hr).
  - End with a clear, low-pressure call to action ("happy to share my portfolio / schedule a call").
  - Do NOT start with "I am a developer looking for work." — that's spam.
  - Sound like a confident, helpful engineer, not a job seeker.
"""
        return self.llm.call(prompt, OpportunityReport)

