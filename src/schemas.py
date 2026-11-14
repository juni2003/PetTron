"""
schemas.py — PetTron Pydantic Data Schemas
==========================================
All structured data models used across PetTron's pipeline.

Part A schemas: CodeAnalysisResult   — what the LLM returns for repo auditing
Part B schemas: OpportunityDraft     — a single ranked opportunity + reply draft
                OpportunityReport    — the full top-3 report
Run schemas:    RunSummary           — recorded in state/run_state.json

These schemas are the "contract" between PetTron's LLM calls and the rest of
the code. The LLM is instructed to respond ONLY with JSON matching these shapes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Code Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class NewFile(BaseModel):
    """
    An entirely new file the bot should create in the target repo
    (e.g., a missing __init__.py, a basic test file, a .gitignore).
    """
    path: str = Field(description="Relative path from repo root, e.g. 'tests/test_utils.py'")
    content: str = Field(description="Full file content to write")


class CodeAnalysisResult(BaseModel):
    """
    Structured output from the LLM after analysing a set of source files.

    If issue_found is False, all other fields will be None/empty and the
    bot skips PR creation for this repo.
    """
    issue_found: bool = Field(
        description="True if a real, fixable issue was found. False if the code looks fine."
    )
    category: Optional[Literal["security", "bug", "performance", "error_handling"]] = Field(
        default=None,
        description="Category of the issue found."
    )
    severity: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="How critical the issue is."
    )
    target_file: Optional[str] = Field(
        default=None,
        description="Relative path to the file that needs patching, e.g. 'src/scraper.py'."
    )
    search_block: Optional[str] = Field(
        default=None,
        description=(
            "The EXACT block of existing lines to find and replace. "
            "Must match the file content character-for-character including indentation. "
            "Keep this as small and precise as possible."
        )
    )
    replace_block: Optional[str] = Field(
        default=None,
        description="The new fixed code that replaces search_block."
    )
    new_files: list[NewFile] = Field(
        default_factory=list,
        description="Optional list of brand-new files to create (e.g., missing test files)."
    )
    pr_title: Optional[str] = Field(
        default=None,
        description="Short, descriptive PR title. E.g. 'fix: add error handling in scraper.py'"
    )
    pr_description: Optional[str] = Field(
        default=None,
        description=(
            "Full Markdown PR body explaining: "
            "## What Changed, ## Why It's Needed, ## How It Works, ## Testing"
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Internal chain-of-thought explaining why this issue was chosen. Not shown in PR."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Opportunity Scout
# ═══════════════════════════════════════════════════════════════════════════════

class OpportunityDraft(BaseModel):
    """
    A single ranked freelance/help-wanted opportunity with a ready-to-copy reply.
    """
    issue_url: str = Field(description="Full URL of the GitHub issue or discussion.")
    issue_title: str = Field(description="Title of the issue.")
    repo_name: str = Field(description="e.g. 'owner/repo'")
    match_score: int = Field(
        ge=1, le=10,
        description="1-10 score: how well this opportunity matches the developer's profile."
    )
    match_reason: str = Field(
        description="One sentence explaining why this is a strong match."
    )
    reply_draft: str = Field(
        description=(
            "A concise, non-spammy, value-first reply draft. "
            "Must: acknowledge their specific problem, mention 1 relevant project with a metric, "
            "state availability and pricing clearly. Must NOT sound like a generic template."
        )
    )


class OpportunityReport(BaseModel):
    """Top 3 opportunities selected and drafted by the LLM."""
    opportunities: list[OpportunityDraft] = Field(
        description="Exactly 3 (or fewer if less were found) top-matched opportunities."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run State
# ═══════════════════════════════════════════════════════════════════════════════

class PRRecord(BaseModel):
    """A record of a PR opened during a run, stored in run_state.json."""
    repo: str
    pr_url: str
    pr_title: str
    category: str
    severity: str
    opened_at: str  # ISO datetime string


class RunSummary(BaseModel):
    """
    One entry in the run_history list in state/run_state.json.
    Written at the end of every daily run.
    """
    run_date: str           # YYYY-MM-DD
    repos_audited: list[str]
    prs_opened: list[PRRecord]
    opportunities_found: int
    llm_used: str           # "gemini" or "groq-fallback"
    duration_seconds: float
    errors: list[str]       # Any non-fatal errors logged during the run

