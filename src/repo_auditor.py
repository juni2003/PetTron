"""
repo_auditor.py — PetTron Part A: Repository Auditor
======================================================
Responsible for:
  1. Reading config/repos.yaml and state/run_state.json to decide WHICH
     repos to audit today (rotation-based, 1-2 per run).
  2. Skipping repos that already have an open bot/fix-* PR.
  3. Fetching source file contents from GitHub via PyGithub API
     (no local clone needed for reading — clean & fast).
  4. Filtering out noise (node_modules, binaries, lockfiles, >50KB files).
  5. Sending the filtered file content to the LLM for analysis.
  6. Returning a CodeAnalysisResult for each audited repo.
"""

import base64
import logging
import os
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Optional

import yaml
from github import Github, GithubException
from github.Repository import Repository

from src.llm_client import LLMClient
from src.schemas import CodeAnalysisResult

logger = logging.getLogger(__name__)

# ── File filtering constants ──────────────────────────────────────────────────

# Directory names to skip entirely (prune the whole subtree)
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", "dist", "build", ".cache", "coverage",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# File extensions considered binary or irrelevant
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav", ".pdf", ".zip", ".tar", ".gz",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".class", ".exe", ".dll", ".so",
    ".db", ".sqlite", ".sqlite3",
}

# Specific filenames to always skip
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock",
    "pnpm-lock.yaml", "Pipfile.lock", "composer.lock",
    ".DS_Store", "Thumbs.db",
}

# Source file extensions to prioritise for analysis
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".sh", ".yaml", ".yml", ".toml", ".env.example",
    ".md", ".txt",
}

MAX_FILE_SIZE_BYTES = 50_000   # 50 KB
MAX_FILES_PER_ANALYSIS = 10    # Max files sent to LLM in one call


class RepoAuditor:
    """
    Orchestrates the daily repo audit cycle for Part A.

    Usage (from main.py):
        auditor = RepoAuditor(gh_client, llm_client, repos_config, run_state)
        results = auditor.run()
        # results → list of (repo_name, CodeAnalysisResult)
    """

    def __init__(
        self,
        gh: Github,
        llm: LLMClient,
        repos_config: dict,
        run_state: dict,
    ) -> None:
        self.gh = gh
        self.llm = llm
        self.repos_config = repos_config
        self.run_state = run_state
        self.skipped_repos: list[str] = []
        # repo_name -> error message, for audits that never produced a verdict.
        # Kept separate from the results so the digest can say "could not audit"
        # instead of the misleading "no issues found".
        self.failed_repos: dict[str, str] = {}

    def run(self) -> list[tuple[str, dict, CodeAnalysisResult]]:
        """
        Main entry point. Returns a list of tuples:
            (repo_name, repo_config_dict, CodeAnalysisResult)

        Only repos where result.issue_found == True are interesting —
        the caller (main.py) will pass those to verifier.py and pr_manager.py.
        """
        repos_to_audit = self._select_repos()
        results = []

        for repo_cfg in repos_to_audit:
            repo_name = repo_cfg["name"]
            logger.info(f"[Auditor] Starting audit of: {repo_name}")
            try:
                result = self._audit_repo(repo_cfg)
                results.append((repo_name, repo_cfg, result))
                # Mark this repo as checked today
                repo_cfg["last_checked"] = date.today().isoformat()
                logger.info(
                    f"[Auditor] {repo_name} → issue_found={result.issue_found}"
                    + (f", category={result.category}" if result.issue_found else "")
                )
            except Exception as exc:
                logger.error(f"[Auditor] Failed to audit {repo_name}: {exc}")
                self.failed_repos[repo_name] = str(exc) or exc.__class__.__name__
                # Advance the rotation anyway: a repo the LLM cannot audit must
                # not pin today's slot forever.
                repo_cfg["last_checked"] = date.today().isoformat()
                # Return a "no issue" result so the run continues; main.py
                # surfaces the failure from self.failed_repos.
                results.append((
                    repo_name,
                    repo_cfg,
                    CodeAnalysisResult(issue_found=False, reasoning=f"Audit error: {exc}"),
                ))

        return results

    # ── Private methods ────────────────────────────────────────────────────────

    def _select_repos(self) -> list[dict]:
        """
        Pick 1-2 repos to audit today using round-robin rotation.

        Repos are sorted by last_checked date (oldest/None first).
        Already-skipped repos (open bot PR) are excluded.
        """
        all_repos: list[dict] = self.repos_config.get("repositories", [])
        per_run: int = self.repos_config.get("rotation", {}).get("repos_per_run", 2)
        skip_if_open_pr: bool = self.repos_config.get("rotation", {}).get(
            "skip_if_open_bot_pr", True
        )

        # Sort by last_checked: None (never checked) first, then oldest date first
        def sort_key(r: dict):
            lc = r.get("last_checked")
            return lc if lc else "0000-00-00"

        sorted_repos = sorted(all_repos, key=sort_key)

        selected = []
        for repo_cfg in sorted_repos:
            if len(selected) >= per_run:
                break
            if skip_if_open_pr and self._has_open_bot_pr(repo_cfg["name"]):
                logger.info(f"[Auditor] Skipping {repo_cfg['name']} — open bot PR exists.")
                self.skipped_repos.append(repo_cfg["name"])
                continue
            selected.append(repo_cfg)

        logger.info(f"[Auditor] Selected {len(selected)} repo(s) for today: "
                    f"{[r['name'] for r in selected]}")
        return selected

    def _has_open_bot_pr(self, repo_name: str) -> bool:
        """Return True if the repo already has an open PR from this bot."""
        try:
            repo: Repository = self.gh.get_repo(repo_name)
            open_prs = repo.get_pulls(state="open")
            for pr in open_prs:
                if pr.head.ref.startswith("bot/fix-"):
                    logger.info(f"[Auditor] Found open bot PR in {repo_name}: #{pr.number}")
                    return True
        except GithubException as exc:
            logger.warning(f"[Auditor] Could not check PRs for {repo_name}: {exc}")
        return False

    def _audit_repo(self, repo_cfg: dict) -> CodeAnalysisResult:
        """Fetch files, filter them, build a prompt, and call the LLM."""
        repo_name = repo_cfg["name"]
        branch = repo_cfg.get("branch", "main")

        repo: Repository = self.gh.get_repo(repo_name)
        files_content = self._fetch_source_files(repo, branch)

        if not files_content:
            logger.warning(f"[Auditor] No source files found in {repo_name}.")
            return CodeAnalysisResult(
                issue_found=False,
                reasoning="No suitable source files found after filtering.",
            )

        prompt = self._build_analysis_prompt(repo_name, files_content)
        result: CodeAnalysisResult = self.llm.call(prompt, CodeAnalysisResult)
        return result

    def _fetch_source_files(
        self, repo: Repository, branch: str
    ) -> dict[str, str]:
        """
        Walk the repo tree via GitHub API and return a dict of
        {relative_path: file_content} for files that pass all filters.

        Caps at MAX_FILES_PER_ANALYSIS to stay within LLM token limits.
        """
        try:
            branch_ref = repo.get_branch(branch)
            tree = repo.get_git_tree(sha=branch_ref.commit.sha, recursive=True).tree
        except GithubException as exc:
            raise RuntimeError(f"Could not read repo tree for branch '{branch}': {exc}") from exc

        files_content: dict[str, str] = {}

        for item in tree:
            if item.type != "blob":
                continue
            path = item.path

            # ── Filter checks ───────────────────────────────────────────
            if self._should_skip_path(path):
                continue
            if item.size and item.size > MAX_FILE_SIZE_BYTES:
                logger.debug(f"[Filter] Skipping large file: {path} ({item.size} bytes)")
                continue
            if len(files_content) >= MAX_FILES_PER_ANALYSIS:
                break

            # ── Fetch content ────────────────────────────────────────────
            try:
                file_obj = repo.get_contents(path, ref=branch)
                # GitHub API returns base64-encoded content
                content = base64.b64decode(file_obj.content).decode("utf-8", errors="replace")

                # Binary file check: look for null bytes
                if "\x00" in content[:1024]:
                    logger.debug(f"[Filter] Skipping binary file: {path}")
                    continue

                files_content[path] = content
            except Exception as exc:
                logger.debug(f"[Filter] Could not read {path}: {exc}")
                continue

        logger.info(f"[Auditor] Collected {len(files_content)} file(s) for analysis.")
        return files_content

    @staticmethod
    def _should_skip_path(path: str) -> bool:
        """Return True if this file path should be excluded from analysis."""
        parts = PurePosixPath(path).parts

        # Skip if any path component is a blacklisted directory
        for part in parts[:-1]:  # All except the filename itself
            if part in SKIP_DIRS:
                return True

        filename = parts[-1]
        suffix = PurePosixPath(filename).suffix.lower()

        if filename in SKIP_FILENAMES:
            return True
        if suffix in SKIP_EXTENSIONS:
            return True
        # Only analyse known source file types
        if suffix and suffix not in SOURCE_EXTENSIONS:
            return True

        return False

    @staticmethod
    def _build_analysis_prompt(repo_name: str, files: dict[str, str]) -> str:
        """
        Build the full prompt sent to the LLM for code analysis.
        """
        files_section = ""
        for path, content in files.items():
            files_section += f"\n\n### FILE: {path}\n```\n{content[:3000]}\n```"
            # Truncate very long files to 3000 chars to protect token budget

        return f"""You are PetTron, an expert code review AI agent. Your task is to audit
the following source files from the GitHub repository "{repo_name}" and identify
ONE high-value, clearly fixable issue.

FOCUS ON finding real issues in this priority order:
  1. SECURITY: SQL injection, hardcoded secrets, unsafe deserialization, missing auth checks
  2. BUG: Logic errors, unhandled exceptions, off-by-one errors, wrong return values
  3. ERROR_HANDLING: Missing try/except blocks around I/O, network, or file operations
  4. PERFORMANCE: Obvious N+1 queries, unnecessary re-computation inside loops

RULES:
  - Only report issue_found=true if you find a REAL, specific, fixable issue.
  - Do NOT report style nits, missing docstrings, or personal preference changes.
  - The search_block must be an EXACT copy of the lines you want to replace
    (copy-paste from the file content above, preserve indentation exactly).
  - The replace_block must be a complete, correct drop-in replacement.
  - If you find nothing meaningful, return issue_found=false.
  - pr_description must use Markdown with sections:
    ## 🔍 What Was Found
    ## 🛠️ What Was Changed
    ## ✅ Why This Fix Is Safe

SOURCE FILES TO REVIEW:
{files_section}
"""

