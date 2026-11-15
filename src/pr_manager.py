"""
pr_manager.py — PetTron Pull Request Manager
=============================================
Responsible for:
  1. Creating a new branch on the target repo via PyGithub API (no git push needed).
  2. Committing the patched file content (and any new files) to that branch.
  3. Opening a detailed Pull Request against the repo's main branch.
  4. Adding labels: "ai-bot" + the issue category (e.g. "bug", "security").
  5. Returning the PR URL for inclusion in the daily digest.

Uses PyGithub's API-based file update (no local git required for the commit step),
which is clean, authenticated automatically via GH_PAT, and works in GitHub Actions.
"""

import logging
from datetime import datetime, timezone

from github import Github, GithubException, UnknownObjectException
from github.Repository import Repository

from src.schemas import CodeAnalysisResult, PRRecord

logger = logging.getLogger(__name__)

# Labels that will be created on the target repo if they don't exist
BOT_LABEL = "ai-bot"
LABEL_COLORS = {
    "ai-bot":         "0075ca",
    "security":       "d73a4a",
    "bug":            "e4e669",
    "performance":    "cfd3d7",
    "error_handling": "a2eeef",
}


class PRManager:
    """
    Handles branch creation, file commits, and PR opening for Part A.

    Usage (from main.py):
        pr_mgr = PRManager(gh_client)
        pr_record = pr_mgr.create_pr(
            repo_name, branch_name, analysis, patched_content
        )
    """

    def __init__(self, gh: Github) -> None:
        self.gh = gh

    def create_pr(
        self,
        repo_name: str,
        branch_name: str,
        analysis: CodeAnalysisResult,
        patched_content: str,
    ) -> PRRecord:
        """
        Create a branch, commit changes, open a PR, and return a PRRecord.

        Args:
            repo_name:       e.g. "juni2003/CrawlX-Data-Scrapping-Project"
            branch_name:     e.g. "bot/fix-20261115-missing-error-handling"
            analysis:        The validated CodeAnalysisResult from the LLM.
            patched_content: Final content of the patched target file.

        Returns:
            PRRecord with the PR URL, title, category, etc.
        """
        repo: Repository = self.gh.get_repo(repo_name)

        # 1. Create the branch from the tip of the repo's default branch
        self._create_branch(repo, branch_name, base_branch=repo.default_branch)

        # 2. Commit the patched file
        self._commit_file(repo, branch_name, analysis.target_file, patched_content)

        # 3. Commit any brand-new files the LLM created (e.g. missing tests)
        for new_file in analysis.new_files:
            self._create_new_file(repo, branch_name, new_file.path, new_file.content)

        # 4. Ensure the labels exist on the target repo
        self._ensure_labels(repo, analysis.category)

        # 5. Open the Pull Request
        pr = self._open_pr(repo, branch_name, analysis)

        logger.info(f"[PRManager] ✅ PR opened: {pr.html_url}")

        return PRRecord(
            repo=repo_name,
            pr_url=pr.html_url,
            pr_title=analysis.pr_title or "PetTron automated fix",
            category=analysis.category or "general",
            severity=analysis.severity or "low",
            opened_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _create_branch(self, repo: Repository, branch_name: str, base_branch: str) -> None:
        """Create a new branch from the tip of base_branch."""
        try:
            base_ref = repo.get_branch(base_branch)
            sha = base_ref.commit.sha
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=sha)
            logger.info(f"[PRManager] Created branch: {branch_name} (from {base_branch}@{sha[:7]})")
        except GithubException as exc:
            if "Reference already exists" in str(exc):
                logger.warning(f"[PRManager] Branch {branch_name} already exists — reusing.")
            else:
                raise RuntimeError(f"Failed to create branch {branch_name}: {exc}") from exc

    def _commit_file(
        self,
        repo: Repository,
        branch_name: str,
        file_path: str,
        new_content: str,
    ) -> None:
        """
        Update an existing file in the branch via GitHub API.
        PyGithub's update_file() needs the current file's SHA.
        """
        try:
            current = repo.get_contents(file_path, ref=branch_name)
            repo.update_file(
                path=file_path,
                message=f"fix: PetTron automated patch — {file_path}",
                content=new_content,
                sha=current.sha,
                branch=branch_name,
            )
            logger.info(f"[PRManager] Committed patch to: {file_path}")
        except GithubException as exc:
            raise RuntimeError(f"Failed to commit {file_path}: {exc}") from exc

    def _create_new_file(
        self,
        repo: Repository,
        branch_name: str,
        file_path: str,
        content: str,
    ) -> None:
        """
        Create a brand-new file in the branch. If the path already exists on the
        branch (e.g. the LLM picked a name that already exists), update it
        instead of failing the whole PR.
        """
        try:
            try:
                existing = repo.get_contents(file_path, ref=branch_name)
            except UnknownObjectException:
                existing = None

            if existing is not None:
                repo.update_file(
                    path=file_path,
                    message=f"chore: PetTron updates {file_path}",
                    content=content,
                    sha=existing.sha,
                    branch=branch_name,
                )
                logger.info(f"[PRManager] Updated existing file: {file_path}")
            else:
                repo.create_file(
                    path=file_path,
                    message=f"feat: PetTron creates {file_path}",
                    content=content,
                    branch=branch_name,
                )
                logger.info(f"[PRManager] Created new file: {file_path}")
        except GithubException as exc:
            logger.warning(f"[PRManager] Could not commit {file_path}: {exc}")

    def _ensure_labels(self, repo: Repository, category: str | None) -> None:
        """Create the required labels on the target repo if they don't exist."""
        labels_needed = [BOT_LABEL]
        if category and category in LABEL_COLORS:
            labels_needed.append(category)

        for label_name in labels_needed:
            try:
                repo.get_label(label_name)
            except UnknownObjectException:
                # Label doesn't exist — create it
                color = LABEL_COLORS.get(label_name, "ededed")
                try:
                    repo.create_label(name=label_name, color=color)
                    logger.info(f"[PRManager] Created label '{label_name}' on {repo.full_name}")
                except GithubException as exc:
                    logger.warning(f"[PRManager] Could not create label '{label_name}': {exc}")

    def _open_pr(
        self,
        repo: Repository,
        branch_name: str,
        analysis: CodeAnalysisResult,
    ):
        """Open the Pull Request and assign labels."""
        # Build a clean PR title with emoji based on category
        emoji_map = {
            "security":       "🔐",
            "bug":            "🐛",
            "performance":    "⚡",
            "error_handling": "🛡️",
        }
        emoji = emoji_map.get(analysis.category or "", "🤖")
        title = f"{emoji} {analysis.pr_title or 'PetTron automated fix'}"

        # Get the repo's default branch
        base_branch = repo.default_branch

        pr = repo.create_pull(
            title=title,
            body=self._build_pr_body(analysis),
            head=branch_name,
            base=base_branch,
        )

        # Add labels (only ones we know exist — unknown categories are skipped)
        labels_to_add = [BOT_LABEL]
        if analysis.category and analysis.category in LABEL_COLORS:
            labels_to_add.append(analysis.category)
        try:
            pr.add_to_labels(*labels_to_add)
        except GithubException as exc:
            logger.warning(f"[PRManager] Could not add labels to PR: {exc}")

        return pr

    @staticmethod
    def _build_pr_body(analysis: CodeAnalysisResult) -> str:
        """Build the full PR body in Markdown."""
        severity_badge = {
            "high":   "🔴 High",
            "medium": "🟡 Medium",
            "low":    "🟢 Low",
        }.get(analysis.severity or "", "⚪ Unknown")

        body = f"""<!-- PetTron Automated PR — Do NOT merge without review -->

> ⚠️ **This PR was opened automatically by [PetTron](https://github.com/juni2003/PetTron).**
> Please review the changes carefully before merging. Never auto-merge.

---

**Severity:** {severity_badge}
**Category:** `{analysis.category}`
**Target File:** `{analysis.target_file}`

---

{analysis.pr_description or "_No description provided by the AI._"}

---

<details>
<summary>🤖 AI Reasoning (internal)</summary>

```
{analysis.reasoning or "N/A"}
```

</details>

---
*Opened by PetTron Bot • [View Bot Repo](https://github.com/juni2003/PetTron)*
"""
        return body

