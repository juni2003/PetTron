"""
verifier.py — PetTron Patch Verifier
======================================
Responsible for:
  1. Applying the LLM's search_block → replace_block patch safely to a local
     clone of the target repository.
  2. Running the repo's configured verify_cmd (e.g. pytest, compileall, npm lint).
  3. If verification fails, sending the error back to the LLM for ONE correction attempt.
  4. If the second attempt also fails → abort cleanly (no PR opened).
  5. Returning the final patched file content (ready for pr_manager.py to commit).

This file never touches .github/workflows/ or .env files (enforced by FORBIDDEN_PATHS).
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from github import Github
    from github.Repository import Repository
    from src.llm_client import LLMClient
from src.schemas import CodeAnalysisResult

logger = logging.getLogger(__name__)

# ── Safety: never patch these paths, no matter what the LLM says ─────────────
FORBIDDEN_PATHS = {
    ".github/workflows",
    ".env",
    ".env.local",
    ".env.production",
    "secrets",
}

MAX_VERIFY_TIMEOUT_SECS = 120   # Kill hung verify commands after 2 minutes


class PatchError(Exception):
    """Raised when a patch cannot be applied (search_block not found)."""


class VerificationError(Exception):
    """Raised when verify_cmd fails and cannot be corrected."""


class Verifier:
    """
    Applies patches and verifies correctness for Part A.

    Usage (from main.py):
        verifier = Verifier(gh_client, llm_client)
        final_content, branch_name = verifier.apply_and_verify(
            repo_name, repo_cfg, analysis_result
        )
        # If this returns without raising, the patch is safe → open a PR.
        # If it raises VerificationError, abort silently.
    """

    def __init__(self, gh: Any, llm: Any) -> None:
        self.gh = gh
        self.llm = llm

    def apply_and_verify(
        self,
        repo_name: str,
        repo_cfg: dict,
        analysis: CodeAnalysisResult,
    ) -> tuple[str, str]:
        """
        Clone the repo, apply the patch, run verify_cmd.

        Returns:
            (patched_file_content, branch_name)

        Raises:
            VerificationError: if both attempts fail (caller should abort).
        """
        if not analysis.target_file or not analysis.search_block or not analysis.replace_block:
            raise VerificationError("Analysis result is missing patch fields — cannot verify.")

        self._check_forbidden(analysis.target_file)

        # Work in a temp directory that is always cleaned up
        tmpdir = tempfile.mkdtemp(prefix="pettron_")
        try:
            repo_dir = self._clone_repo(repo_name, repo_cfg["branch"], tmpdir)
            return self._attempt_verify(repo_dir, repo_name, repo_cfg, analysis, attempt=1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"[Verifier] Cleaned up temp dir: {tmpdir}")

    # ── Private methods ────────────────────────────────────────────────────────

    def _check_forbidden(self, target_file: str) -> None:
        """Raise VerificationError if the target file is in a forbidden path."""
        for forbidden in FORBIDDEN_PATHS:
            if target_file.startswith(forbidden) or target_file == forbidden:
                raise VerificationError(
                    f"[Safety] Refusing to patch forbidden path: {target_file}"
                )

    def _clone_repo(self, repo_name: str, branch: str, into_dir: str) -> Path:
        """
        Clone the target repo using the GH_PAT for authentication.
        Returns the path to the cloned repo root.
        """
        pat = os.environ["GH_PAT"]
        clone_url = f"https://{pat}@github.com/{repo_name}.git"
        repo_dir = Path(into_dir) / repo_name.split("/")[-1]

        logger.info(f"[Verifier] Cloning {repo_name} (branch={branch}) ...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--branch", branch, clone_url, str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise VerificationError(f"Git clone failed: {result.stderr[:500]}")

        logger.info(f"[Verifier] Cloned to: {repo_dir}")
        return repo_dir

    def _attempt_verify(
        self,
        repo_dir: Path,
        repo_name: str,
        repo_cfg: dict,
        analysis: CodeAnalysisResult,
        attempt: int,
    ) -> tuple[str, str]:
        """
        Apply patch, run verify_cmd. If it fails and attempt==1, ask LLM to fix it.
        """
        target_path = repo_dir / analysis.target_file
        if not target_path.exists():
            raise VerificationError(
                f"Target file not found in cloned repo: {analysis.target_file}"
            )

        # ── Apply the patch ────────────────────────────────────────────────
        try:
            patched_content = self._apply_patch(
                target_path,
                analysis.search_block,
                analysis.replace_block,
            )
        except PatchError as exc:
            raise VerificationError(str(exc))

        # Write any new files the LLM wants to create
        for new_file in analysis.new_files:
            new_path = repo_dir / new_file.path
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(new_file.content, encoding="utf-8")
            logger.info(f"[Verifier] Created new file: {new_file.path}")

        # ── Run verify_cmd ─────────────────────────────────────────────────
        verify_cmd = repo_cfg.get("verify_cmd", "python -m compileall . -q")
        logger.info(f"[Verifier] Running verify_cmd (attempt {attempt}): {verify_cmd}")

        verify_result = subprocess.run(
            verify_cmd,
            shell=True,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=MAX_VERIFY_TIMEOUT_SECS,
        )

        if verify_result.returncode == 0:
            logger.info(f"[Verifier] ✅ Verification passed on attempt {attempt}.")
            # Return the final patched content of the target file
            final_content = target_path.read_text(encoding="utf-8")
            branch_name = self._make_branch_name(analysis)
            return final_content, branch_name

        # ── Verification failed ────────────────────────────────────────────
        error_output = (verify_result.stdout + verify_result.stderr)[:2000]
        logger.warning(f"[Verifier] ❌ Verification failed (attempt {attempt}):\n{error_output}")

        if attempt >= 2:
            raise VerificationError(
                f"Verification failed after self-correction attempt.\n"
                f"Error output:\n{error_output}"
            )

        # ── Self-correction: ask the LLM once to fix its own patch ────────
        logger.info("[Verifier] Asking LLM to self-correct the patch ...")
        corrected = self._ask_llm_to_correct(analysis, error_output)

        if not corrected.issue_found or not corrected.search_block:
            raise VerificationError("LLM could not produce a corrected patch.")

        # Reset target file to original before retry
        self._clone_repo(repo_name, repo_cfg["branch"], str(repo_dir.parent))

        return self._attempt_verify(repo_dir, repo_name, repo_cfg, corrected, attempt=2)

    def _ask_llm_to_correct(
        self, original: CodeAnalysisResult, error_output: str
    ) -> CodeAnalysisResult:
        """Send the error back to the LLM and ask for a corrected patch."""
        prompt = f"""You previously suggested this code fix but the verification command FAILED.

ORIGINAL TARGET FILE: {original.target_file}

YOUR ORIGINAL search_block:
```
{original.search_block}
```

YOUR ORIGINAL replace_block:
```
{original.replace_block}
```

VERIFICATION ERROR OUTPUT:
```
{error_output}
```

Please analyse the error and provide a CORRECTED patch.
- If you cannot fix the error, set issue_found=false.
- The search_block must EXACTLY match the original file content.
- The replace_block must be syntactically correct.
"""
        return self.llm.call(prompt, CodeAnalysisResult)

    @staticmethod
    def _apply_patch(file_path: Path, search_block: str, replace_block: str) -> str:
        """
        Apply a search/replace patch to a file.

        Raises PatchError if the search_block is not found exactly in the file.
        Returns the new file content (also writes it to disk).
        """
        original = file_path.read_text(encoding="utf-8")

        if search_block not in original:
            raise PatchError(
                f"search_block not found verbatim in {file_path.name}. "
                "The LLM may have hallucinated the existing code. Aborting patch."
            )

        patched = original.replace(search_block, replace_block, 1)  # Replace first occurrence only
        file_path.write_text(patched, encoding="utf-8")
        logger.info(f"[Verifier] Patch applied to: {file_path.name}")
        return patched

    @staticmethod
    def _make_branch_name(analysis: CodeAnalysisResult) -> str:
        """
        Generate a branch name like: bot/fix-20261115-missing-error-handling
        Slugifies the pr_title for readability.
        """
        from datetime import date
        import re

        today = date.today().strftime("%Y%m%d")
        title = analysis.pr_title or analysis.category or "patch"
        # Keep only alphanumeric and hyphens, lowercase, max 40 chars
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        return f"bot/fix-{today}-{slug}"

