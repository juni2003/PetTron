"""
llm_client.py — PetTron LLM Client
====================================
Handles all AI calls for PetTron with:
  • Primary:  Google Gemini 2.0 Flash  (1M TPM — ideal for large code analysis)
  • Fallback: Groq Llama 3.1 8B       (30 RPM, 100% free — used if Gemini 429s)

Both are 100% free tier. Gemini is primary because its 1M token-per-minute
context window handles multi-file code analysis without hitting TPM limits.
Groq (8K TPM) is used as a reliable fallback for smaller prompts.

Usage:
    from src.llm_client import LLMClient
    from src.schemas import CodeAnalysisResult

    client = LLMClient()
    result: CodeAnalysisResult = client.analyze_code(files_content, schema=CodeAnalysisResult)
"""

import os
import time
import json
import logging
from typing import Type, TypeVar

from google import genai
from google.genai import types as genai_types
from groq import Groq
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ── Type variable so analyze() can return the correct Pydantic model ──────────
T = TypeVar("T", bound=BaseModel)

# ── Model identifiers ─────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ── Retry / backoff settings ──────────────────────────────────────────────────
MAX_RETRIES       = 3
BASE_BACKOFF_SECS = 5   # seconds; doubles on each retry


class LLMError(Exception):
    """Raised when both Gemini and Groq fail after all retries."""


class LLMClient:
    """
    Unified LLM client for PetTron.

    Instantiate once in main.py and pass around — it lazily initialises
    the Gemini and Groq SDK clients using environment variables.
    """

    def __init__(self) -> None:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        groq_key   = os.environ.get("GROQ_API_KEY", "")

        if not gemini_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        if not groq_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

        self._gemini = genai.Client(api_key=gemini_key)
        self._groq   = Groq(api_key=groq_key)

        logger.info(f"LLMClient ready | primary={GEMINI_MODEL} | fallback={GROQ_MODEL}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(self, prompt: str, schema: Type[T]) -> T:
        """
        Send *prompt* to the LLM and parse the response into *schema*.

        Tries Gemini first (up to MAX_RETRIES), then falls back to Groq.
        Always returns a validated Pydantic instance of *schema*.

        Args:
            prompt: The full prompt string to send.
            schema: A Pydantic BaseModel class to parse the JSON response into.

        Returns:
            A validated instance of *schema*.

        Raises:
            LLMError: If both providers fail after all retries.
        """
        # ── 1. Try Gemini ──────────────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Gemini attempt {attempt}/{MAX_RETRIES} ...")
                raw = self._call_gemini(prompt, schema)
                result = self._parse(raw, schema)
                logger.info("Gemini call succeeded.")
                return result

            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                logger.warning(f"Gemini attempt {attempt} failed: {exc}")

                if attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                    logger.info(f"Backing off {wait}s before retry ...")
                    time.sleep(wait)
                elif is_rate_limit:
                    logger.warning("Gemini rate-limited. Switching to Groq fallback ...")
                    break  # Try Groq
                else:
                    break  # Non-rate-limit error — still try Groq

        # ── 2. Groq fallback ───────────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Groq fallback attempt {attempt}/{MAX_RETRIES} ...")
                raw = self._call_groq(prompt, schema)
                result = self._parse(raw, schema)
                logger.info("Groq fallback succeeded.")
                return result

            except Exception as exc:
                logger.warning(f"Groq attempt {attempt} failed: {exc}")
                if attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                    time.sleep(wait)

        raise LLMError("Both Gemini and Groq failed after all retries. Aborting this task.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_gemini(self, prompt: str, schema: Type[T]) -> str:
        """Call Gemini and return the raw text response."""
        full_prompt = self._wrap_with_json_instruction(prompt, schema)
        response = self._gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        return response.text

    def _call_groq(self, prompt: str, schema: Type[T]) -> str:
        """Call Groq and return the raw text response."""
        full_prompt = self._wrap_with_json_instruction(prompt, schema)
        response = self._groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    @staticmethod
    def _wrap_with_json_instruction(prompt: str, schema: Type[T]) -> str:
        """
        Append a JSON format instruction to the prompt so the LLM knows
        to respond with valid JSON matching the Pydantic schema.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        return (
            f"{prompt}\n\n"
            "---\n"
            "IMPORTANT: Respond with ONLY a valid JSON object. "
            "No markdown, no code fences, no explanation. "
            f"The JSON must conform to this schema:\n{schema_json}"
        )

    @staticmethod
    def _parse(raw_text: str, schema: Type[T]) -> T:
        """
        Parse a raw LLM text response into the given Pydantic schema.
        Strips any accidental markdown code fences before parsing.
        """
        # Strip ```json ... ``` fences if the LLM added them anyway
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Failed to parse LLM response into {schema.__name__}: {exc}\nRaw: {raw_text[:500]}")

