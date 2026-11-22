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
import posixpath
import re
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
    ) -> tuple[str, str, CodeAnalysisResult]:
        """
        Clone the repo, apply the patch, run verify_cmd.

        Returns:
            (patched_file_content, branch_name, analysis_used)

            ``analysis_used`` is the *corrected* analysis when the LLM had to
            fix its own patch, so the caller opens the PR against the file that
            was actually patched.

        Raises:
            VerificationError: if both attempts fail (caller should abort).
        """
        if not analysis.target_file or not analysis.search_block or not analysis.replace_block:
            raise VerificationError("Analysis result is missing patch fields — cannot verify.")

        self._check_forbidden(analysis.target_file)
        # Brand-new files must respect the same safety rules as the target file.
        for new_file in analysis.new_files:
            self._check_forbidden(new_file.path)

        # Work in a temp directory that is always cleaned up
        tmpdir = tempfile.mkdtemp(prefix="pettron_")
        try:
            repo_dir = self._clone_repo(repo_name, repo_cfg["branch"], tmpdir)
            self._assert_baseline_verify(repo_dir, repo_cfg)
            return self._attempt_verify(repo_dir, repo_name, repo_cfg, analysis, attempt=1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"[Verifier] Cleaned up temp dir: {tmpdir}")

    # ── Private methods ────────────────────────────────────────────────────────

    def _check_forbidden(self, target_file: str) -> None:
        """Raise VerificationError if the target file is in a forbidden path."""
        if not target_file or not str(target_file).strip():
            raise VerificationError("[Safety] Refusing to patch an empty path.")

        # Normalise Windows separators, "./" prefixes and ".." traversal so
        # that './.github/workflows/x.yml' and 'src/../.env' cannot slip past.
        raw = str(target_file).replace("\\", "/").strip()
        normalized = posixpath.normpath(raw).lstrip("/").lower()
        if normalized in ("", "."):
            raise VerificationError(f"[Safety] Refusing to patch invalid path: {target_file}")
        if normalized == ".." or normalized.startswith("../"):
            raise VerificationError(
                f"[Safety] Refusing to patch a path outside the repository: {target_file}"
            )

        parts = [p for p in normalized.split("/") if p]

        for forbidden in FORBIDDEN_PATHS:
            f = forbidden.lower()
            if normalized == f or normalized.startswith(f + "/") or f in parts:
                raise VerificationError(
                    f"[Safety] Refusing to patch forbidden path: {target_file}"
                )

    def _clone_repo(self, repo_name: str, branch: str, into_dir: str) -> Path:
        """
        Clone the target repo using the GH_PAT for authentication.
        Returns the path to the cloned repo root.

        The PAT is never written to logs or error messages — git can echo the
        remote URL on failure, so all output is scrubbed with ``_redact``.
        """
        pat = (os.environ.get("GH_PAT") or "").strip()
        if not pat:
            raise VerificationError("GH_PAT is not set — cannot clone the target repo.")

        clone_url = f"https://{pat}@github.com/{repo_name}.git"
        repo_dir = Path(into_dir) / repo_name.split("/")[-1]

        logger.info(f"[Verifier] Cloning {repo_name} (branch={branch}) ...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--branch", branch, clone_url, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise VerificationError(f"Git clone timed out for {repo_name}") from exc

        if result.returncode != 0:
            raise VerificationError(
                f"Git clone failed: {self._redact(result.stderr or result.stdout)[:500]}"
            )

        logger.info(f"[Verifier] Cloned to: {repo_dir}")
        return repo_dir

    @staticmethod
    def _redact(text: str) -> str:
        """Remove any token/credential pattern from text before logging it."""
        if not text:
            return ""
        # Scrub 'user:token@host' and bare long tokens.
        text = re.sub(r"//[^@/\s]+@", "//***@", text)
        text = re.sub(r"(gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,})", "***", text)
        return text

    def _attempt_verify(
        self,
        repo_dir: Path,
        repo_name: str,
        repo_cfg: dict,
        analysis: CodeAnalysisResult,
        attempt: int,
    ) -> tuple[str, str, CodeAnalysisResult]:
        """
        Apply patch, run verify_cmd. If it fails and attempt==1, ask LLM to fix it.
        """
        # Re-check on every attempt: a self-corrected analysis is new LLM output.
        self._check_forbidden(analysis.target_file)

        target_path = repo_dir / analysis.target_file
        if not target_path.exists():
            raise VerificationError(
                f"Target file not found in cloned repo: {analysis.target_file}"
            )

        # ── Apply the patch ────────────────────────────────────────────────
        try:
            self._apply_patch(
                target_path,
                analysis.search_block,
                analysis.replace_block,
            )
        except PatchError as exc:
            if attempt >= 2:
                raise VerificationError(str(exc))

            # The LLM guessed the existing code instead of copying it. Show it
            # the real file so it can re-emit a search_block that exists.
            logger.warning(f"[Verifier] {exc} — asking the LLM to re-read the file ...")
            corrected = self._ask_llm_to_fix_patch(
                analysis, target_path.read_text(encoding="utf-8")
            )
            if not corrected.issue_found or not corrected.search_block:
                raise VerificationError(
                    "LLM could not produce a patch that matches the real file."
                ) from exc

            self._reset_worktree(repo_dir)
            return self._attempt_verify(repo_dir, repo_name, repo_cfg, corrected, attempt=2)

        # Write any new files the LLM wants to create. Skip paths that already
        # exist (they are edits, not new files) and never clobber a directory.
        for new_file in analysis.new_files:
            self._check_forbidden(new_file.path)
            new_path = repo_dir / new_file.path
            if new_path.exists():
                logger.warning(
                    f"[Verifier] Skipping new_file '{new_file.path}' — it already exists."
                )
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(new_file.content, encoding="utf-8")
            logger.info(f"[Verifier] Created new file: {new_file.path}")

        # ── Run verify_cmd ─────────────────────────────────────────────────
        verify_cmd = repo_cfg.get("verify_cmd", "python -m compileall . -q")
        logger.info(f"[Verifier] Running verify_cmd (attempt {attempt}): {verify_cmd}")

        returncode, error_output = self._run_verify(repo_dir, verify_cmd)

        if returncode == 0:
            logger.info(f"[Verifier] ✅ Verification passed on attempt {attempt}.")
            # Return the final patched content of the target file
            final_content = target_path.read_text(encoding="utf-8")
            branch_name = self._make_branch_name(analysis)
            return final_content, branch_name, analysis

        # ── Verification failed ────────────────────────────────────────────
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

        # Roll the working tree back to the pristine clone before retrying.
        # (Re-cloning into the same directory fails because it already exists.)
        self._reset_worktree(repo_dir)

        return self._attempt_verify(repo_dir, repo_name, repo_cfg, corrected, attempt=2)

    @staticmethod
    def _run_verify(repo_dir: Path, verify_cmd: str) -> tuple[int, str]:
        """Run *verify_cmd* in *repo_dir*; return (returncode, combined output)."""
        try:
            result = subprocess.run(
                verify_cmd,
                shell=True,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=MAX_VERIFY_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired:
            return 124, f"verify_cmd timed out after {MAX_VERIFY_TIMEOUT_SECS}s"
        return result.returncode, Verifier._redact((result.stdout + result.stderr)[:2000])

    def _assert_baseline_verify(self, repo_dir: Path, repo_cfg: dict) -> None:
        """
        Fail before patching if verify_cmd already fails on the pristine clone.

        Otherwise an already-broken repo looks like the patch broke it: the run
        logs an empty "Error output:" and the PR is lost for no reason.
        """
        verify_cmd = repo_cfg.get("verify_cmd", "python -m compileall . -q")
        if repo_cfg.get("baseline_verify", True) is False:
            return
        logger.info(f"[Verifier] Checking verify_cmd on the unmodified clone: {verify_cmd}")
        returncode, output = self._run_verify(repo_dir, verify_cmd)
        if returncode != 0:
            tail = output.strip().splitlines()[-1][:200] if output.strip() else "no output"
            raise VerificationError(
                f"verify_cmd fails on the unmodified repo (exit {returncode}), "
                f"so a patch cannot be judged: {verify_cmd} → {tail}"
            )
        logger.info("[Verifier] Baseline verification passed.")

    @staticmethod
    def _reset_worktree(repo_dir: Path) -> None:
        """
        Discard all local changes in *repo_dir* so the next attempt starts from
        the original clone: revert tracked files and remove untracked ones
        (including any new files a previous attempt created).
        """
        for cmd in (
            ["git", "checkout", "--", "."],
            ["git", "clean", "-fd"],
        ):
            result = subprocess.run(
                cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                logger.warning(
                    "[Verifier] Worktree reset command %s failed: %s",
                    cmd,
                    (result.stderr or "").strip()[:200],
                )

    # Keep the prompt small enough for every provider's context window; a
    # truncated read is flagged in the prompt so the model knows to stay safe.
    MAX_FILE_CHARS_FOR_PROMPT = 12000

    def _ask_llm_to_fix_patch(
        self, original: CodeAnalysisResult, file_text: str
    ) -> CodeAnalysisResult:
        """
        Ask the LLM for a new patch when the first one could not be applied.

        The real file content is included because "the search_block does not
        exist" is almost always the model paraphrasing the code it was shown;
        feeding it the verbatim file is what lets it recover.
        """
        excerpt = file_text[: self.MAX_FILE_CHARS_FOR_PROMPT]
        truncation = (
            ""
            if len(file_text) <= self.MAX_FILE_CHARS_FOR_PROMPT
            else f"\n(file truncated after {self.MAX_FILE_CHARS_FOR_PROMPT} characters)"
        )
        prompt = f"""Your patch was rejected: the search_block does not appear in the real file.

TARGET FILE: {original.target_file}

REAL FILE CONTENT (verbatim):
```
{excerpt}{truncation}
```

YOUR search_block (NOT found in the file):
```
{original.search_block}
```

YOUR replace_block:
```
{original.replace_block}
```

Produce the fix again against the file content above.
- Copy search_block character-for-character from the file (same indentation, same quotes).
- Keep it short, but unique enough to appear exactly once.
- If the fix cannot be made safely, set issue_found=false.
"""
        return self.llm.call(prompt, CodeAnalysisResult)

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
    def _match_ignoring_whitespace(original: str, search_block: str) -> tuple[int, int] | None:
        """
        Find *search_block* in *original* comparing stripped lines.

        Returns the (start, end) character offsets of the matched region, or
        None when there is no match or more than one (ambiguous).
        """
        wanted = [line.strip() for line in search_block.strip("\n").splitlines()]
        if not any(wanted):
            return None

        lines = original.splitlines(keepends=True)
        stripped = [line.strip() for line in lines]
        width = len(wanted)
        hits = [
            i
            for i in range(len(stripped) - width + 1)
            if stripped[i : i + width] == wanted
        ]
        if len(hits) != 1:
            return None

        start_line = hits[0]
        start = sum(len(line) for line in lines[:start_line])
        end = start + sum(len(line) for line in lines[start_line : start_line + width])
        return start, end

    # Literals from another language that a model writes into Python by mistake
    # ("agent_initialized = false" compiles fine and only fails at runtime).
    _FOREIGN_LITERAL = re.compile(r"(?<![\w.\"'])=(\s*)(true|false|null|undefined)\b")

    @staticmethod
    def _code_lines(block: str) -> list[str]:
        """
        Return the lines of a block that carry behaviour, not prose.

        Comments and docstring/string-only lines are dropped so a patch that
        only rewords text can be told apart from one that changes logic.
        """
        code = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped[0] in "\"'":
                continue
            code.append(stripped)
        return code

    @classmethod
    def _reject_low_value_patch(
        cls, file_path: Path, search_block: str, replace_block: str
    ) -> None:
        """
        Refuse a patch that would open a PR without changing behaviour.

        Both cases below reached real PRs: one rewrote a docstring and changed
        the wording of a returned string, the other wrote JavaScript's ``false``
        into Python, which ``compileall`` happily accepts.
        """
        added = cls._code_lines(replace_block)

        if not added:
            raise PatchError(
                f"patch for {file_path.name} only touches comments, docstrings or "
                "removes code. Not opening a PR for a patch that changes no behaviour."
            )

        if file_path.suffix == ".py":
            for line in added:
                if cls._FOREIGN_LITERAL.search(line):
                    raise PatchError(
                        f"patch for {file_path.name} uses a JavaScript literal "
                        f"(true/false/null) in Python: {line!r}. "
                        "Python needs True/False/None."
                    )

    @staticmethod
    def _apply_patch(file_path: Path, search_block: str, replace_block: str) -> str:
        """
        Apply a search/replace patch to a file.

        Raises PatchError if the search_block is not found exactly in the file.
        Returns the new file content (also writes it to disk).
        """
        original = file_path.read_text(encoding="utf-8")
        Verifier._reject_low_value_patch(file_path, search_block, replace_block)

        if search_block in original:
            patched = original.replace(search_block, replace_block, 1)  # First occurrence only
        else:
            # Models often reproduce the code with a different indentation or
            # trailing whitespace. Accept that only when exactly one region
            # matches; anything ambiguous is a real mismatch.
            span = Verifier._match_ignoring_whitespace(original, search_block)
            if span is None:
                raise PatchError(
                    f"search_block not found verbatim in {file_path.name}. "
                    "The LLM may have hallucinated the existing code. Aborting patch."
                )
            start, end = span
            tail = replace_block
            if end < len(original) and not tail.endswith("\n"):
                tail += "\n"
            patched = original[:start] + tail + original[end:]
            logger.info(
                f"[Verifier] search_block matched ignoring indentation in {file_path.name}."
            )
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

        today = date.today().strftime("%Y%m%d")
        title = analysis.pr_title or analysis.category or "patch"
        # Keep only alphanumeric and hyphens, lowercase, max 40 chars
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        return f"bot/fix-{today}-{slug}"

