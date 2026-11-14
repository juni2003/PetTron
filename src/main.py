"""
main.py — PetTron Entry Point & Orchestrator
=============================================
This is the single script invoked by GitHub Actions every day.

Pipeline:
  1. Load config (repos.yaml, profile.json, run_state.json)
  2. Init GitHub + LLM clients
  3. PART A — Repo Auditor loop:
       repo_auditor → verifier → pr_manager
  4. PART B — Opportunity Scout:
       opportunity_scout → drafts
  5. NOTIFIER — daily GitHub Issue digest
  6. Save updated run_state.json back to disk (workflow commits it)

Exit codes:
  0 — Successful run (even if no PRs were opened)
  1 — Fatal startup error (missing secrets, bad config, etc.)
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import yaml
from github import Github

from src.llm_client import LLMClient
from src.notifier import Notifier
from src.opportunity_scout import OpportunityScout
from src.pr_manager import PRManager
from src.repo_auditor import RepoAuditor
from src.schemas import PRRecord, RunSummary
from src.verifier import VerificationError, Verifier

# ── Paths (relative to repo root, where GitHub Actions runs from) ─────────────
CONFIG_DIR    = Path("config")
REPOS_YAML    = CONFIG_DIR / "repos.yaml"
PROFILE_JSON  = CONFIG_DIR / "profile.json"
STATE_FILE    = Path("state") / "run_state.json"


def setup_logging() -> None:
    """Configure structured logging for GitHub Actions (no colour codes)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def check_required_secrets() -> None:
    """Fail fast if any required environment variable is missing."""
    required = ["GH_PAT", "GEMINI_API_KEY", "GROQ_API_KEY", "PRIVATE_REPO_FOR_LOG"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        logging.critical(f"Missing required secrets: {missing}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    logger = logging.getLogger("pettron.main")
    start_time = time.monotonic()

    logger.info("=" * 60)
    logger.info("🤖 PetTron starting up ...")
    logger.info("=" * 60)

    # ── 0. Validate environment ────────────────────────────────────────────
    check_required_secrets()

    gh_pat            = os.environ["GH_PAT"]
    private_repo_name = os.environ["PRIVATE_REPO_FOR_LOG"]

    # ── 1. Load configs ────────────────────────────────────────────────────
    logger.info("Loading configs ...")
    repos_config = load_yaml(REPOS_YAML)
    run_state    = load_json(STATE_FILE)

    # ── 2. Init clients ────────────────────────────────────────────────────
    logger.info("Initialising GitHub and LLM clients ...")
    gh  = Github(gh_pat)
    llm = LLMClient()

    # ── 3. PART A — Repo Auditor ───────────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("PART A: Repo Auditor starting ...")
    logger.info("─" * 40)

    prs_opened:      list[PRRecord] = []
    repos_no_issue:  list[str]      = []
    repos_skipped:   list[str]      = []
    run_errors:      list[str]      = []
    llm_used:        str            = "gemini-2.0-flash"

    auditor  = RepoAuditor(gh, llm, repos_config, run_state)
    verifier = Verifier(gh, llm)
    pr_mgr   = PRManager(gh)

    audit_results = auditor.run()
    repos_skipped = auditor.skipped_repos

    for repo_name, repo_cfg, analysis in audit_results:
        if not analysis.issue_found:
            logger.info(f"[Main] {repo_name}: no issue found — skipping PR.")
            repos_no_issue.append(repo_name)
            continue

        logger.info(
            f"[Main] {repo_name}: issue found "
            f"({analysis.category}, {analysis.severity}) → running verifier ..."
        )

        try:
            patched_content, branch_name = verifier.apply_and_verify(
                repo_name, repo_cfg, analysis
            )
            logger.info(f"[Main] Verification passed. Opening PR on {repo_name} ...")
            pr_record = pr_mgr.create_pr(
                repo_name=repo_name,
                branch_name=branch_name,
                analysis=analysis,
                patched_content=patched_content,
            )
            prs_opened.append(pr_record)
            logger.info(f"[Main] ✅ PR opened: {pr_record.pr_url}")

        except VerificationError as exc:
            msg = f"Verification failed for {repo_name}: {exc}"
            logger.warning(f"[Main] ⚠️  {msg}")
            run_errors.append(msg)

        except Exception as exc:
            msg = f"Unexpected error processing {repo_name}: {exc}"
            logger.error(f"[Main] ❌ {msg}")
            run_errors.append(msg)

    # ── 4. PART B — Opportunity Scout ─────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("PART B: Opportunity Scout starting ...")
    logger.info("─" * 40)

    from src.schemas import OpportunityReport
    opportunity_report = OpportunityReport(opportunities=[])

    try:
        scout = OpportunityScout(gh, llm, PROFILE_JSON)
        opportunity_report = scout.run()
        logger.info(
            f"[Main] Scout found {len(opportunity_report.opportunities)} top opportunit"
            f"{'y' if len(opportunity_report.opportunities)==1 else 'ies'}."
        )
    except Exception as exc:
        msg = f"Opportunity Scout failed: {exc}"
        logger.error(f"[Main] ❌ {msg}")
        run_errors.append(msg)

    # ── 5. Send daily digest ───────────────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("NOTIFIER: Sending daily digest ...")
    logger.info("─" * 40)

    duration_secs = time.monotonic() - start_time

    try:
        notifier = Notifier(gh, private_repo_name)
        digest_url = notifier.send_digest(
            prs_opened=prs_opened,
            repos_skipped=repos_skipped,
            repos_no_issue=repos_no_issue,
            opportunity_report=opportunity_report,
            run_errors=run_errors,
            duration_secs=duration_secs,
            llm_used=llm_used,
        )
        logger.info(f"[Main] ✅ Digest sent: {digest_url}")
    except Exception as exc:
        logger.error(f"[Main] ❌ Failed to send digest: {exc}")

    # ── 6. Update run_state.json ───────────────────────────────────────────
    today = date.today().isoformat()
    run_summary = {
        "run_date":           today,
        "repos_audited":      [r for r, _, _ in audit_results],
        "prs_opened":         [p.model_dump() for p in prs_opened],
        "opportunities_found": len(opportunity_report.opportunities),
        "llm_used":           llm_used,
        "duration_seconds":   round(duration_secs, 1),
        "errors":             run_errors,
    }

    run_state["last_run_date"]     = today
    run_state["repos_checked_today"] = [r for r, _, _ in audit_results]
    run_state["prs_opened_today"]  = [p.model_dump() for p in prs_opened]
    run_state.setdefault("run_history", []).append(run_summary)
    # Keep only last 30 days of history to avoid state file bloat
    run_state["run_history"] = run_state["run_history"][-30:]

    save_json(STATE_FILE, run_state)
    save_yaml(REPOS_YAML, repos_config)   # Saves updated last_checked dates

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ PetTron run complete in {duration_secs:.1f}s")
    logger.info(f"   PRs opened: {len(prs_opened)}")
    logger.info(f"   Opportunities drafted: {len(opportunity_report.opportunities)}")
    logger.info(f"   Errors: {len(run_errors)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
