"""
llm_client.py — PetTron LLM Client
====================================
Handles all AI calls for PetTron with:
  • Primary:  Google Gemini 2.0 Flash  (1M TPM — ideal for large code analysis)
  • Fallback: Groq Llama 3.1 8B       (30 RPM, 100% free — used if Gemini fails)

Both are 100% free tier. Gemini is primary because its 1M token-per-minute
context window handles multi-file code analysis without hitting TPM limits.
Groq (8K TPM) is used as a reliable fallback for smaller prompts.

Structured output
-----------------
Gemini is asked for structured JSON via ``response_schema=<Pydantic class>``.
The SDK then converts the model into the API-native ``responseSchema`` shape
(nested objects inlined, no ``$defs`` / ``$ref`` / ``additionalProperties``).

Do NOT pass a raw Pydantic ``model_json_schema()`` to ``response_json_schema``:
that emits ``$defs`` / ``$ref`` / ``additionalProperties``, which the Gemini
API rejects with ``400 INVALID_ARGUMENT`` on *every* attempt. That single
mistake silently disabled both Part A and Part B.

Usage:
    from src.llm_client import LLMClient
    from src.schemas import CodeAnalysisResult

    client = LLMClient()
    result: CodeAnalysisResult = client.call(prompt, schema=CodeAnalysisResult)
    print(client.last_provider)   # e.g. "gemini:gemini-2.0-flash"
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Type, TypeVar

from google import genai
from google.genai import types as genai_types
from groq import Groq
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ── Type variable so call() can return the correct Pydantic model ─────────────
T = TypeVar("T", bound=BaseModel)

# ── Model identifiers (all overridable via environment / GitHub Secrets) ──────
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-flash-latest"]
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# ── Retry / backoff settings ──────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_BACKOFF_SECS = 5   # seconds; doubles on each retry

# Substrings that mean "this model itself is gone", not a transient error.
_MODEL_UNAVAILABLE_MARKERS = (
    "404",
    "not_found",
    "not found",
    "no longer available",
    "is not supported",
    "does not exist",
    "unsupported model",
    "was not found",
)

# Substrings that mean "retrying the same request cannot help" (bad key, bad
# request body, schema rejected by the API). Retrying these just burns ~100s
# inside the workflow before the real error finally surfaces.
_NON_RETRYABLE_MARKERS = (
    "400",
    "invalid_argument",
    "invalid argument",
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "api key expired",
    "unauthenticated",
    "permission_denied",
    "permission denied",
    "401",
    "403",
)


class LLMError(Exception):
    """Raised when both Gemini and Groq fail after all retries."""


def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated env var into a list, falling back to *default*."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_model_unavailable(exc: BaseException) -> bool:
    """True when the error indicates the *model* is gone (try another model)."""
    text = str(exc).lower()
    return any(marker in text for marker in _MODEL_UNAVAILABLE_MARKERS)


def _is_non_retryable(exc: BaseException) -> bool:
    """True when repeating the same request is pointless (bad key / bad body)."""
    text = str(exc).lower()
    return any(marker in text for marker in _NON_RETRYABLE_MARKERS)


class LLMClient:
    """
    Unified LLM client for PetTron.

    Instantiate once in main.py and pass around — it lazily initialises
    the Gemini and Groq SDK clients using environment variables.
    """

    def __init__(self) -> None:
        gemini_key = (
            os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        ).strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()

        if not gemini_key:
            raise EnvironmentError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set.")
        if not groq_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

        # Ensure both are set in environment for the SDKs
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key

        # Primary model first, then any configured fallbacks.
        self.gemini_models = _dedupe(
            [os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)]
            + _env_list("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACK_MODELS)
        )
        self.groq_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        self._gemini = genai.Client(api_key=gemini_key)
        self._groq = Groq(api_key=groq_key)

        # Which provider actually answered, e.g. "gemini:gemini-2.0-flash".
        self.last_provider: str | None = None

        logger.info(
            "LLMClient ready | gemini=%s (fallbacks=%s) | groq=%s",
            self.gemini_models[0],
            self.gemini_models[1:] or "none",
            self.groq_model,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(self, prompt: str, schema: Type[T]) -> T:
        """
        Send *prompt* to the LLM and parse the response into *schema*.

        Tries each configured Gemini model, then falls back to Groq.
        Always returns a validated Pydantic instance of *schema*, and sets
        ``self.last_provider`` to whichever provider succeeded.

        Args:
            prompt: The full prompt string to send.
            schema: A Pydantic BaseModel class to parse the JSON response into.

        Returns:
            A validated instance of *schema*.

        Raises:
            LLMError: If both providers fail after all retries.
        """
        self.last_provider = None
        errors: list[str] = []

        # ── 1. Gemini (primary, across configured models) ──────────────────
        for index, model in enumerate(self.gemini_models):
            try_next_model = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info("[LLM] Gemini model=%s attempt %d/%d ...", model, attempt, MAX_RETRIES)
                    raw = self._call_gemini(prompt, schema, model=model)
                    result = self._parse(raw, schema)
                    self.last_provider = f"gemini:{model}"
                    logger.info("[LLM] Gemini (%s) call succeeded.", model)
                    return result

                except Exception as exc:
                    detail = f"gemini:{model} attempt {attempt}: {type(exc).__name__}: {exc}"
                    errors.append(detail)
                    logger.warning("[LLM] %s", detail)

                    if _is_model_unavailable(exc):
                        if index + 1 < len(self.gemini_models):
                            logger.warning(
                                "[LLM] Gemini model '%s' is unavailable — trying '%s' instead.",
                                model,
                                self.gemini_models[index + 1],
                            )
                            try_next_model = True
                        else:
                            logger.warning(
                                "[LLM] Gemini model '%s' is unavailable and no fallbacks remain.",
                                model,
                            )
                        break

                    if _is_non_retryable(exc):
                        logger.error(
                            "[LLM] Non-retryable Gemini error — not retrying '%s'.", model
                        )
                        break

                    if attempt < MAX_RETRIES:
                        wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                        logger.info("[LLM] Backing off %ss before retry ...", wait)
                        time.sleep(wait)

            if not try_next_model:
                break  # attempts exhausted for this model — move on to Groq

        logger.warning("[LLM] Gemini exhausted. Switching to Groq fallback (%s) ...", self.groq_model)

        # ── 2. Groq fallback ───────────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("[LLM] Groq model=%s attempt %d/%d ...", self.groq_model, attempt, MAX_RETRIES)
                raw = self._call_groq(prompt, schema)
                result = self._parse(raw, schema)
                self.last_provider = f"groq:{self.groq_model}"
                logger.info("[LLM] Groq fallback call succeeded.")
                return result

            except Exception as exc:
                detail = f"groq:{self.groq_model} attempt {attempt}: {type(exc).__name__}: {exc}"
                errors.append(detail)
                logger.warning("[LLM] %s", detail)

                if _is_non_retryable(exc):
                    logger.error("[LLM] Non-retryable Groq error — not retrying.")
                    break

                if attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                    logger.info("[LLM] Backing off %ss before retry ...", wait)
                    time.sleep(wait)

        raise LLMError(
            "Both Gemini and Groq failed after all retries. "
            f"Last errors: {' | '.join(errors[-6:])}"
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_gemini_config(schema: Type[T]) -> genai_types.GenerateContentConfig:
        """
        Build the Gemini generation config with a schema the API accepts.

        Passing the Pydantic *class* (not ``model_json_schema()``) lets the SDK
        emit the native ``responseSchema`` — nested objects inlined, no
        ``$defs`` / ``$ref`` / ``additionalProperties``.
        """
        return genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )

    def _call_gemini(self, prompt: str, schema: Type[T], model: str | None = None) -> str:
        """Call Gemini with native structured JSON output."""
        config = self._build_gemini_config(schema)
        response = self._gemini.models.generate_content(
            model=model or self.gemini_models[0],
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def _call_groq(self, prompt: str, schema: Type[T]) -> str:
        """Call Groq with JSON mode and the schema described in the prompt."""
        full_prompt = (
            f"{prompt}\n\n"
            "---\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        response = self._groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that outputs only valid JSON "
                        "conforming strictly to the requested schema."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4096,
        )
        return (response.choices[0].message.content or "")

    @staticmethod
    def _parse(raw_text: str, schema: Type[T]) -> T:
        """
        Parse a raw LLM text response into the given Pydantic schema.
        Handles fences, markdown commentary, and extracts JSON objects cleanly.
        """
        if raw_text is None or not str(raw_text).strip():
            raise ValueError(f"Empty LLM response — cannot parse into {schema.__name__}.")

        text = str(raw_text).strip()

        # 1. Strip markdown fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()

        # 2. Extract substring between first '{' and last '}'
        if not text.startswith("{") and "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse LLM response into {schema.__name__}: {exc}\n"
                f"Raw: {str(raw_text)[:500]}"
            ) from exc

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"LLM response did not match {schema.__name__}: {exc}\n"
                f"Raw: {str(raw_text)[:500]}"
            ) from exc


# ── Standalone diagnostic: python -m src.llm_client ───────────────────────────
# Runs a real call against each provider and prints the *exact* error, so a
# flaky key or a retired model is obvious in a few seconds instead of after a
# ~2-minute retry loop inside the daily run.

SELFTEST_PROMPT = (
    'Return a JSON object with a single key "opportunities". '
    "Its value must be an empty array."
)


def _selftest() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from src.schemas import OpportunityReport

    try:
        client = LLMClient()
    except Exception as exc:
        print(f"❌ LLMClient init failed: {type(exc).__name__}: {exc}")
        return 1

    gemini_ok = False
    for model in client.gemini_models:
        try:
            raw = client._call_gemini(SELFTEST_PROMPT, OpportunityReport, model=model)
            client._parse(raw, OpportunityReport)
            print(f"✅ Gemini '{model}': OK")
            gemini_ok = True
            break
        except Exception as exc:
            print(f"❌ Gemini '{model}': {type(exc).__name__}: {exc}")

    groq_ok = False
    try:
        raw = client._call_groq(SELFTEST_PROMPT, OpportunityReport)
        client._parse(raw, OpportunityReport)
        print(f"✅ Groq '{client.groq_model}': OK")
        groq_ok = True
    except Exception as exc:
        print(f"❌ Groq '{client.groq_model}': {type(exc).__name__}: {exc}")

    print("-" * 50)
    if gemini_ok and groq_ok:
        print("✅ Both providers healthy.")
        return 0
    if gemini_ok or groq_ok:
        print("⚠️  Only one provider healthy — the run will still work via fallback.")
        return 0
    print("❌ Both providers failed. Fix the keys/models above before the next run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
