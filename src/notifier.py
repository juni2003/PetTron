"""
notifier.py — PetTron Daily Digest via GitHub Private Issue
============================================================
Responsible for:
  Creating one GitHub Issue per day in the private log repository
  (defined by the PRIVATE_REPO_FOR_LOG secret).

  The Issue contains:
    - Part A summary: list of repos audited + any PRs opened with direct links
    - Part B summary: top 3 opportunities with ready-to-copy reply drafts
    - Run metadata: duration, LLM used, any errors

  Since the private log repo is watched by the owner, GitHub Mobile will
  send a push notification immediately when the Issue is created.

  No Telegram, no email, no extra credentials required beyond GH_PAT.
"""

import logging
import os
from datetime import datetime, timezone

from github import Github, GithubException, UnknownObjectException
from github.Repository import Repository

from src.schemas import OpportunityReport, PRRecord

logger = logging.getLogger(__name__)


class Notifier:
    """
    Creates the daily digest Issue in the private log repository.

    Usage (from main.py):
        notifier = Notifier(gh_client, private_repo_name)
        notifier.send_digest(
            prs_opened=prs_opened,
            repos_skipped=repos_skipped,
            repos_no_issue=repos_no_issue,
            repos_failed=repos_failed,
            opportunity_report=opportunity_report,
            run_errors=run_errors,
            duration_secs=duration_secs,
            llm_used=llm_used,
        )
    """

    def __init__(self, gh: Github, private_repo_name: str) -> None:
        self.gh = gh
        self.private_repo_name = private_repo_name

    def send_digest(
        self,
        prs_opened: list[PRRecord],
        repos_skipped: list[str],
        repos_no_issue: list[str],
        opportunity_report: OpportunityReport,
        run_errors: list[str],
        duration_secs: float,
        llm_used: str,
        repos_failed: dict[str, str] | None = None,
    ) -> str:
        """
        Create the daily digest Issue and return its URL.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        title = f"🤖 PetTron Daily Digest — {today}"
        body = self._build_body(
            today=today,
            prs_opened=prs_opened,
            repos_skipped=repos_skipped,
            repos_no_issue=repos_no_issue,
            repos_failed=repos_failed,
            opportunity_report=opportunity_report,
            run_errors=run_errors,
            duration_secs=duration_secs,
            llm_used=llm_used,
        )

        try:
            repo: Repository = self._get_or_create_repo()
            existing = self._find_existing_digest(repo, title)
            if existing is not None:
                # A manual re-run on the same day: append instead of spamming
                # a second digest issue.
                comment = existing.create_comment(body)
                logger.info(f"[Notifier] ✅ Appended to today's digest: {comment.html_url}")
                return existing.html_url
            issue = repo.create_issue(title=title, body=body)
            logger.info(f"[Notifier] ✅ Digest issue created: {issue.html_url}")
            return issue.html_url
        except Exception as exc:
            logger.error(f"[Notifier] Failed to create digest issue: {exc}")
            return ""

    @staticmethod
    def _find_existing_digest(repo: Repository, title: str):
        """Return the open digest issue for *title*, if one already exists."""
        try:
            for issue in repo.get_issues(state="open"):
                if issue.pull_request is None and issue.title == title:
                    return issue
        except Exception as exc:
            logger.warning(f"[Notifier] Could not check for an existing digest: {exc}")
        return None

    def _get_or_create_repo(self) -> Repository:
        """
        Get the target log repo. If it does not exist (404), automatically
        create it as a private repository. If creation fails, fall back to
        the main PetTron repository so the issue is never lost.
        """
        # Normalize target repo name (handle 'pettron-logs' without username)
        target = self.private_repo_name.strip()
        if "/" not in target:
            try:
                username = self.gh.get_user().login
                target = f"{username}/{target}"
            except Exception:
                target = f"juni2003/{target}"

        try:
            repo = self.gh.get_repo(target)
            # Access an attribute to force checking existence
            _ = repo.id
            return repo
        except (UnknownObjectException, GithubException) as exc:
            logger.info(
                f"[Notifier] Repo '{target}' not found or inaccessible: {exc}. "
                "Attempting to create it automatically as a private repository..."
            )
            try:
                repo_name_only = target.split("/")[-1]
                user = self.gh.get_user()
                repo = user.create_repo(
                    name=repo_name_only,
                    private=True,
                    description="PetTron automated daily digest logs",
                    auto_init=True,
                )
                logger.info(f"[Notifier] ✅ Created private repo: {repo.html_url}")
                return repo
            except Exception as create_exc:
                logger.warning(
                    f"[Notifier] Could not auto-create repo '{target}': {create_exc}."
                )

        fallback_name = os.environ.get("GITHUB_REPOSITORY", "juni2003/PetTron")
        logger.info(f"[Notifier] Falling back to log issue in '{fallback_name}'")
        return self.gh.get_repo(fallback_name)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_body(
        today: str,
        prs_opened: list[PRRecord],
        repos_skipped: list[str],
        repos_no_issue: list[str],
        opportunity_report: OpportunityReport,
        run_errors: list[str],
        duration_secs: float,
        llm_used: str,
        repos_failed: dict[str, str] | None = None,
    ) -> str:
        """Build the full Markdown body of the digest Issue."""

        # ── PART A section ──────────────────────────────────────────────────
        part_a_lines = ["## 📋 Part A — Repo Auditor\n"]

        if prs_opened:
            part_a_lines.append(f"**{len(prs_opened)} PR(s) opened today:**\n")
            for pr in prs_opened:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(pr.severity, "⚪")
                part_a_lines.append(
                    f"- {severity_icon} **[{pr.pr_title}]({pr.pr_url})**\n"
                    f"  - Repo: `{pr.repo}` | Category: `{pr.category}` | Severity: `{pr.severity}`"
                )
        else:
            part_a_lines.append("No PRs opened today.")

        if repos_no_issue:
            part_a_lines.append(
                "\n**Repos audited (no issues found):** "
                + ", ".join(f"`{r}`" for r in repos_no_issue)
            )

        if repos_failed:
            part_a_lines.append(
                f"\n**Repos that could NOT be audited ({len(repos_failed)}):**"
            )
            for name, reason in repos_failed.items():
                first_line = (reason or "").strip().splitlines()[0][:160]
                part_a_lines.append(f"- `{name}` — {first_line}")

        if repos_skipped:
            part_a_lines.append(
                "\n**Repos skipped (open bot PR exists):** "
                + ", ".join(f"`{r}`" for r in repos_skipped)
            )

        part_a_section = "\n".join(part_a_lines)

        # ── PART B section ──────────────────────────────────────────────────
        part_b_lines = ["\n---\n\n## 💼 Part B — Opportunity Scout\n"]

        opps = opportunity_report.opportunities if opportunity_report else []

        if opps:
            part_b_lines.append(
                f"Found **{len(opps)} top opportunit{'y' if len(opps)==1 else 'ies'}** "
                "to explore. Review drafts below and post manually if they look good.\n"
            )
            for i, opp in enumerate(opps, 1):
                score_stars = "⭐" * min(opp.match_score, 10)
                part_b_lines.append(f"""
### 🎯 Opportunity {i} — Match Score: {opp.match_score}/10 {score_stars}

**Issue:** [{opp.issue_title}]({opp.issue_url})
**Repo:** `{opp.repo_name}`
**Why it matches:** {opp.match_reason}

<details>
<summary>📝 Click to expand Reply Draft (copy & post manually)</summary>

```
{opp.reply_draft}
```

</details>
""")
        else:
            part_b_lines.append("No strong opportunities found in today's search. Try again tomorrow.")

        part_b_section = "\n".join(part_b_lines)

        # ── Errors section ──────────────────────────────────────────────────
        errors_section = ""
        if run_errors:
            errors_section = "\n---\n\n## ⚠️ Errors / Warnings\n\n"
            errors_section += "\n".join(f"- `{e}`" for e in run_errors)

        # ── Metadata footer ──────────────────────────────────────────────────
        mins = int(duration_secs // 60)
        secs = int(duration_secs % 60)
        footer = f"""
---

## ⚙️ Run Metadata

| Field | Value |
|---|---|
| Date | {today} |
| Duration | {mins}m {secs}s |
| LLM Used | `{llm_used}` |
| PRs Opened | {len(prs_opened)} |
| Opportunities Drafted | {len(opps)} |

---
*Generated by [PetTron Bot](https://github.com/juni2003/PetTron) 🤖*
*⚠️ All PRs and reply drafts require your manual review before use.*
"""

        return part_a_section + part_b_section + errors_section + footer

