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
from datetime import date
from pathlib import Path

# Ensure repo root is always on sys.path so 'from src.xxx' works anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    """Fail fast with clean diagnostic message if any required secret is missing."""
    required = ["GH_PAT", "GEMINI_API_KEY", "GROQ_API_KEY", "PRIVATE_REPO_FOR_LOG"]
    missing = [k for k in required if not (os.environ.get(k) or "").strip()]
    if missing:
        print(f"\n❌ [PetTron] CRITICAL: Missing required GitHub Secrets: {missing}", flush=True)
        print("Please check: Repo -> Settings -> Secrets and variables -> Actions", flush=True)
        print("Make sure each secret is created as a Repository secret and is not blank.\n", flush=True)
        sys.exit(1)

    # A secret pasted with surrounding quotes or a trailing newline looks valid
    # to this check but fails authentication later (401/400). Clean it up and say so.
    for key in required:
        raw = os.environ.get(key, "")
        cleaned = raw.strip().strip('"').strip("'").strip()
        if cleaned != raw:
            os.environ[key] = cleaned
            print(
                f"⚠️  [PetTron] Secret '{key}' had stray whitespace/quotes — cleaned automatically.",
                flush=True,
            )


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
    audit_results:   list           = []

    auditor  = RepoAuditor(gh, llm, repos_config, run_state)
    verifier = Verifier(gh, llm)
    pr_mgr   = PRManager(gh)

    # A failure in Part A must never stop Part B or the daily digest.
    try:
        audit_results = auditor.run()
        repos_skipped = auditor.skipped_repos
    except Exception as exc:
        msg = f"Repo Auditor failed: {exc}"
        logger.error(f"[Main] ❌ {msg}")
        run_errors.append(msg)

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
            patched_content, branch_name, final_analysis = verifier.apply_and_verify(
                repo_name, repo_cfg, analysis
            )
            logger.info(f"[Main] Verification passed. Opening PR on {repo_name} ...")
            pr_record = pr_mgr.create_pr(
                repo_name=repo_name,
                branch_name=branch_name,
                # Use the analysis that was actually applied (it may have been
                # self-corrected by the LLM after a failed verify run).
                analysis=final_analysis,
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
    # Report the model that actually answered (e.g. "gemini:gemini-2.0-flash"
    # or "groq:llama-3.1-8b-instant") instead of a hard-coded guess.
    llm_used = llm.last_provider or "not-called"

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
        if digest_url:
            logger.info(f"[Main] ✅ Digest sent: {digest_url}")
        else:
            logger.warning(
                "[Main] ⚠️  Digest could not be posted — see the Notifier log above."
            )
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
    # Replace any earlier entry for today (manual re-runs), drop older
    # duplicates left by previous runs, and keep the last 30 days.
    by_date = {
        h["run_date"]: h
        for h in run_state.get("run_history", [])
        if isinstance(h, dict) and h.get("run_date")
    }
    by_date[today] = run_summary
    run_state["run_history"] = [by_date[d] for d in sorted(by_date)][-30:]

    save_json(STATE_FILE, run_state)
    save_yaml(REPOS_YAML, repos_config)   # Saves updated last_checked dates

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ PetTron run complete in {duration_secs:.1f}s")
    logger.info(f"   PRs opened: {len(prs_opened)}")
    logger.info(f"   Opportunities drafted: {len(opportunity_report.opportunities)}")
    logger.info(f"   Errors: {len(run_errors)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"\n❌ [PetTron] Fatal unhandled exception in main:\n{exc}\n", flush=True)
        traceback.print_exc()
        sys.exit(1)
