"""
tests/test_llm.py — Unit & diagnostic tests for src/llm_client.py
==================================================================
Covers the two things that broke the Opportunity Scout (and silently
disabled Part A) in production:

  1. The Gemini request must use a schema the API actually accepts.
     Passing ``response_json_schema`` a raw Pydantic ``model_json_schema()``
     emits ``$defs`` / ``$ref`` / ``additionalProperties``, which the Gemini
     API rejects with 400 INVALID_ARGUMENT on every retry — so both providers
     appeared to fail and the scout aborted.  We assert the exact request
     payload the SDK would send instead.

  2. ``_parse()`` must turn messy LLM output (fences, commentary, arrays,
     empty bodies) into a validated model, and must never crash with an
     opaque AttributeError.

The live-API tests at the bottom only run when real keys are present in the
environment, so the suite stays green offline.

Run:
    pytest tests/ -v
    # live:
    GEMINI_API_KEY=... GROQ_API_KEY=... pytest tests/test_llm.py -v
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from google.genai._api_client import BaseApiClient, SdkHttpResponse
from google.genai.models import Models

from src.llm_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    GEMINI_API_VERSION,
    LLMClient,
    LLMError,
)
from src.schemas import CodeAnalysisResult, OpportunityReport

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_gemini_response_body() -> str:
    """A minimal, valid Gemini response envelope."""
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "{}"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
    )


def capture_gemini_generation_config(schema) -> dict:
    """
    Instantiate an LLMClient (with dummy keys), intercept the SDK's HTTP
    request, and return the ``generationConfig`` it would have sent.
    """
    captured: dict = {}

    def fake_request(self, http_method, path, request_dict, http_options=None):
        captured["request"] = request_dict
        return SdkHttpResponse(headers={}, body=_fake_gemini_response_body())

    env = {"GEMINI_API_KEY": "dummy-gemini", "GROQ_API_KEY": "dummy-groq"}
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(BaseApiClient, "request", fake_request),
    ):
        client = LLMClient()
        client._call_gemini("say hi", schema)

    return captured["request"]["generationConfig"]


class TestGeminiSchemaIsApiCompatible(unittest.TestCase):
    """
    Regression tests for the exact bug that broke production.

    The Gemini API accepts the native ``responseSchema`` shape (inlined
    objects, no JSON-Schema ``$ref``) and rejects raw JSON-Schema documents
    containing ``$defs`` / ``$ref``.
    """

    def test_uses_response_schema_not_response_json_schema(self):
        gen = capture_gemini_generation_config(OpportunityReport)
        self.assertIn("responseSchema", gen)
        self.assertNotIn(
            "responseJsonSchema",
            gen,
            "Raw JSON-Schema (with $defs/$ref) is rejected by the Gemini API.",
        )

    def test_nested_models_are_inlined_without_defs_or_refs(self):
        gen = capture_gemini_generation_config(OpportunityReport)
        encoded = json.dumps(gen["responseSchema"])
        self.assertNotIn("$defs", encoded)
        self.assertNotIn("$ref", encoded)
        self.assertNotIn("additionalProperties", encoded)

    def test_nested_array_items_are_present(self):
        gen = capture_gemini_generation_config(OpportunityReport)
        schema = gen["responseSchema"]
        self.assertEqual(schema["type"], "OBJECT")
        self.assertIn("opportunities", schema["properties"])
        items = schema["properties"]["opportunities"]["items"]
        self.assertIn("reply_draft", items["properties"])

    def test_code_analysis_schema_also_compatible(self):
        gen = capture_gemini_generation_config(CodeAnalysisResult)
        encoded = json.dumps(gen["responseSchema"])
        self.assertNotIn("$defs", encoded)
        self.assertNotIn("$ref", encoded)


class TestGeminiApiVersion(unittest.TestCase):
    """
    Structured output (``responseMimeType: application/json`` + schema) is only
    served on ``v1beta``. The google-genai SDK defaults to ``v1``, which answers
    ``400 INVALID_ARGUMENT: JSON mode is not enabled for api version v1`` — the
    exact error that made every Gemini call fail in production.
    """

    def test_client_pins_v1beta(self):
        self.assertEqual(GEMINI_API_VERSION, "v1beta")
        env = {"GEMINI_API_KEY": "dummy-gemini", "GROQ_API_KEY": "dummy-groq"}
        with mock.patch.dict(os.environ, env, clear=False):
            client = LLMClient()
        self.assertEqual(client._gemini._api_client._http_options.api_version, "v1beta")

    def test_request_asks_for_json_mime_type(self):
        gen = capture_gemini_generation_config(OpportunityReport)
        self.assertEqual(gen.get("responseMimeType"), "application/json")


class TestRetiredModelIds(unittest.TestCase):
    """
    Google and Groq both retire model ids with no notice. A 404 must not be
    retried three times and must not end the run when the key still works.
    """

    def _client(self, gemini=None, groq=None) -> LLMClient:
        env = {"GEMINI_API_KEY": "dummy-gemini", "GROQ_API_KEY": "dummy-groq"}
        with mock.patch.dict(os.environ, env, clear=False):
            client = LLMClient()
        if gemini is not None:
            client.gemini_models = list(gemini)
        if groq is not None:
            client.groq_models = list(groq)
        return client

    def test_gemini_404_lists_models_and_uses_a_live_one(self):
        client = self._client(gemini=["gemini-2.0-flash", "gemini-flash-latest"])
        listed = [
            SimpleNamespace(name="models/gemini-9-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
        ]
        seen: list[str] = []

        def fake_call(prompt, schema, model=None):
            seen.append(model)
            if model in ("gemini-2.0-flash", "gemini-flash-latest"):
                raise RuntimeError(
                    "404 NOT_FOUND: This model models/gemini-2.0-flash is no longer available."
                )
            return json.dumps({"opportunities": []})

        with (
            mock.patch.object(client, "_call_gemini", side_effect=fake_call),
            mock.patch.object(Models, "list", return_value=listed),
            mock.patch("src.llm_client.time.sleep"),
        ):
            result = client.call("prompt", OpportunityReport)

        self.assertEqual(result.opportunities, [])
        self.assertEqual(client.last_provider, "gemini:gemini-9-flash")
        # One attempt per retired id, then the discovered one.
        self.assertEqual(seen, ["gemini-2.0-flash", "gemini-flash-latest", "gemini-9-flash"])
        self.assertIn("gemini-9-flash", client.gemini_models)
        self.assertNotIn("text-embedding-004", client.gemini_models)

    def test_gemini_listing_is_cached_across_calls(self):
        client = self._client(gemini=["gone"])
        with (
            mock.patch.object(client, "_call_gemini", side_effect=RuntimeError("404 NOT_FOUND")),
            mock.patch.object(Models, "list", return_value=[]) as list_models,
            mock.patch("src.llm_client.time.sleep"),
        ):
            client._try_gemini("p", OpportunityReport, errors=[])
            client._try_gemini("p", OpportunityReport, errors=[])

        # Discovery is cached: one listing per client, however many failures.
        self.assertEqual(list_models.call_count, 1)

    def test_groq_unknown_model_skips_retries_and_tries_next(self):
        client = self._client(gemini=[], groq=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        seen: list[str] = []

        def fake_call(prompt, schema, model=None):
            seen.append(model)
            if model == "llama-3.1-8b-instant":
                raise RuntimeError(
                    "Error code: 404 - model_not_found: The model "
                    "`llama-3.1-8b-instant` does not exist or you do not have access to it."
                )
            return json.dumps({"opportunities": []})

        with (
            mock.patch.object(client, "_call_groq", side_effect=fake_call),
            mock.patch("src.llm_client.time.sleep"),
        ):
            result = client.call("prompt", OpportunityReport)

        self.assertEqual(seen, ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        self.assertEqual(result.opportunities, [])
        self.assertEqual(client.last_provider, "groq:llama-3.3-70b-versatile")

    def test_groq_lists_models_when_every_configured_one_is_gone(self):
        client = self._client(gemini=[], groq=["dead-model"])
        listing = SimpleNamespace(
            data=[
                SimpleNamespace(id="dead-model", active=False),
                SimpleNamespace(id="whisper-large-v3", active=True),
                SimpleNamespace(id="llama-live", active=True),
            ]
        )

        def fake_call(prompt, schema, model=None):
            if model == "dead-model":
                raise RuntimeError("404 model_not_found: does not exist")
            return json.dumps({"opportunities": []})

        with (
            mock.patch.object(client, "_call_groq", side_effect=fake_call),
            mock.patch.object(client._groq, "models") as models,
            mock.patch("src.llm_client.time.sleep"),
        ):
            models.list.return_value = listing
            client.call("prompt", OpportunityReport)

        self.assertIn("llama-live", client.groq_models)
        self.assertNotIn("whisper-large-v3", client.groq_models)


class TestCredentials(unittest.TestCase):
    def test_duplicate_google_api_key_is_dropped(self):
        env = {
            "GEMINI_API_KEY": "g",
            "GROQ_API_KEY": "q",
            "GOOGLE_API_KEY": "same-secret-again",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            LLMClient()
            self.assertNotIn("GOOGLE_API_KEY", os.environ)

    def test_blank_model_env_vars_fall_back_to_defaults(self):
        """An unset GitHub Actions `vars` entry expands to "" — never a model id."""
        env = {
            "GEMINI_API_KEY": "g",
            "GROQ_API_KEY": "q",
            "GEMINI_MODEL": "",
            "GROQ_MODEL": "   ",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            client = LLMClient()
        self.assertEqual(client.gemini_models[0], DEFAULT_GEMINI_MODEL)
        self.assertEqual(client.groq_models[0], DEFAULT_GROQ_MODEL)


class TestParse(unittest.TestCase):
    """_parse() must be forgiving of real-world LLM output and never crash."""

    def test_plain_json(self):
        raw = json.dumps({"opportunities": []})
        result = LLMClient._parse(raw, OpportunityReport)
        self.assertEqual(result.opportunities, [])

    def test_json_in_markdown_fence(self):
        raw = "```json\n" + json.dumps({"opportunities": []}) + "\n```"
        result = LLMClient._parse(raw, OpportunityReport)
        self.assertEqual(result.opportunities, [])

    def test_json_with_leading_commentary(self):
        raw = "Sure! Here you go:\n" + json.dumps({"opportunities": []}) + "\nHope it helps."
        result = LLMClient._parse(raw, OpportunityReport)
        self.assertEqual(result.opportunities, [])

    def test_whitespace_only_raises_value_error(self):
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                LLMClient._parse(bad, OpportunityReport)

    def test_none_raises_value_error_not_attribute_error(self):
        with self.assertRaises(ValueError):
            LLMClient._parse(None, OpportunityReport)

    def test_malformed_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            LLMClient._parse("{not valid json", OpportunityReport)

    def test_schema_mismatch_raises_value_error(self):
        raw = json.dumps({"opportunities": [{"issue_url": "x"}]})  # missing fields
        with self.assertRaises(ValueError):
            LLMClient._parse(raw, OpportunityReport)


class TestFallbackAndReporting(unittest.TestCase):
    """The client must fall back to Groq, report the provider used, and
    surface the real underlying error when both providers fail."""

    def _client(self) -> LLMClient:
        env = {"GEMINI_API_KEY": "dummy-gemini", "GROQ_API_KEY": "dummy-groq"}
        with mock.patch.dict(os.environ, env, clear=False):
            return LLMClient()

    def test_groq_fallback_used_and_reported(self):
        client = self._client()
        good = json.dumps({"opportunities": []})

        with (
            mock.patch.object(client, "_call_gemini", side_effect=RuntimeError("gemini down")),
            mock.patch.object(client, "_call_groq", return_value=good),
            mock.patch("src.llm_client.time.sleep"),  # no real backoff
        ):
            result = client.call("prompt", OpportunityReport)

        self.assertEqual(result.opportunities, [])
        self.assertTrue(client.last_provider.startswith("groq:"))

    def test_both_fail_raises_llmerror_with_details(self):
        client = self._client()
        # Pin one model per provider so the call count is deterministic.
        client.gemini_models = ["g-primary"]
        client.groq_models = ["q-primary"]

        with (
            mock.patch.object(client, "_call_gemini", side_effect=RuntimeError("gemini boom")),
            mock.patch.object(client, "_call_groq", side_effect=RuntimeError("groq boom")),
            mock.patch("src.llm_client.time.sleep"),
            self.assertRaises(LLMError) as ctx,
        ):
            client.call("prompt", OpportunityReport)

        message = str(ctx.exception)
        self.assertIn("gemini boom", message)
        self.assertIn("groq boom", message)
        # The message must name the model, not just repeat one provider.
        self.assertIn("gemini:g-primary", message)
        self.assertIn("groq:q-primary", message)

    def test_non_retryable_error_fails_fast_instead_of_retrying(self):
        """A bad key / rejected schema must not burn ~2 minutes of backoff."""
        client = self._client()
        calls = {"n": 0}

        def bad_request(prompt, schema, *args, **kwargs):
            calls["n"] += 1
            raise RuntimeError(
                "400 INVALID_ARGUMENT: API key not valid. Please pass a valid API key."
            )

        with (
            mock.patch.object(client, "_call_gemini", side_effect=bad_request),
            mock.patch.object(client, "_call_groq", side_effect=bad_request),
            mock.patch("src.llm_client.time.sleep"),
            self.assertRaises(LLMError),
        ):
            client.call("prompt", OpportunityReport)

        # 1 attempt on Gemini + 1 on Groq — no pointless retries.
        self.assertEqual(calls["n"], 2)

    def test_transient_error_still_uses_all_retries(self):
        """A 503 is retryable: all attempts must run before giving up."""
        client = self._client()
        client.gemini_models = ["g-primary"]
        client.groq_models = ["q-primary"]
        calls = {"n": 0}

        def flaky(prompt, schema, *args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("503 UNAVAILABLE: model overloaded")

        with (
            mock.patch.object(client, "_call_gemini", side_effect=flaky),
            mock.patch.object(client, "_call_groq", side_effect=flaky),
            mock.patch("src.llm_client.time.sleep"),
            self.assertRaises(LLMError),
        ):
            client.call("prompt", OpportunityReport)

        self.assertEqual(calls["n"], 6)  # 3 Gemini attempts + 3 Groq attempts

    def test_model_unavailable_falls_through_to_next_gemini_model(self):
        client = self._client()
        client.gemini_models = ["retired-model", "working-model"]
        good = json.dumps({"opportunities": []})
        calls = {"n": 0}

        def flaky(prompt, schema, model=None):
            calls["n"] += 1
            if model == "retired-model":
                raise RuntimeError("404 NOT_FOUND: model not found")
            return good

        with (
            mock.patch.object(client, "_call_gemini", side_effect=flaky),
            mock.patch("src.llm_client.time.sleep"),
        ):
            result = client.call("prompt", OpportunityReport)

        self.assertEqual(result.opportunities, [])
        self.assertEqual(calls["n"], 2)  # one try each model, no pointless retries
        self.assertIn("working-model", client.last_provider)


@unittest.skipUnless(os.environ.get("GEMINI_API_KEY"), "GEMINI_API_KEY not set")
class TestLiveGemini(unittest.TestCase):
    """Only runs when a real key is available — reports the exact error if not."""

    def test_live_gemini_structured_output(self):
        client = LLMClient()
        prompt = (
            'Return a JSON object with an "opportunities" array containing no items. '
            "The array must be present."
        )
        report = client.call(prompt, OpportunityReport)
        self.assertIsInstance(report, OpportunityReport)


@unittest.skipUnless(os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY not set")
class TestLiveGroq(unittest.TestCase):
    """Only runs when a real key is available — reports the exact error if not."""

    def test_live_groq_structured_output(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
            client = LLMClient()
        prompt = (
            'Return a JSON object with an "opportunities" array containing no items. '
            "The array must be present."
        )
        # Exercise Groq directly so the diagnostic is fast and unambiguous.
        raw = client._call_groq(prompt, OpportunityReport)
        report = client._parse(raw, OpportunityReport)
        self.assertIsInstance(report, OpportunityReport)


if __name__ == "__main__":
    unittest.main()
