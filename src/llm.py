"""Cached, deterministic Anthropic client for the audit (SPEC.md section 7).

    call(system, user, model, max_tokens=1024) -> dict

Every call is cached on sha256(system + user + model) under cache/llm/, so a
re-run costs nothing. Responses are parsed as JSON; a parse failure gets one
repair retry before raising.

Model IDs below are the current Sonnet and Haiku models as documented by
Anthropic (Claude Sonnet 5, Claude Haiku 4.5). Do not date-suffix them.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.config import LLM_CACHE_DIR

# --- models ------------------------------------------------------------------
# SPEC.md section 7: judge on the current Sonnet, generation on the current Haiku.
# Verified against Anthropic's model catalog; these aliases are complete as-is.
MODEL_JUDGE = "claude-sonnet-5"         # Claude Sonnet 5
MODEL_GENERATION = "claude-haiku-4-5"   # Claude Haiku 4.5

# SPEC.md section 1 asks for temperature=0 on every call. That is not
# implementable: the anthropic 1.x SDK has REMOVED temperature, top_p and top_k
# from Messages.create() entirely, and the current models reject non-default
# sampling parameters with a 400. Determinism therefore comes from the on-disk
# cache — the same (system, user, model) always returns the same bytes — which
# is what the spec actually needs ("re-runs must cost nothing").
_SAMPLING_PARAMS_AVAILABLE = False

# These models run adaptive thinking when `thinking` is omitted, which would
# consume the max_tokens budget and truncate the JSON response. Disable it
# explicitly: the judge and the generator both emit short structured JSON.
_THINKING_ON_BY_DEFAULT = {
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
}

MAX_RETRIES = 2        # retries on rate limit / 5xx, on top of the first attempt
BACKOFF_BASE = 2.0     # seconds: 2s, then 4s
REPAIR_SUFFIX = "\n\nYour previous response was not valid JSON. Respond with JSON only."

_client: anthropic.Anthropic | None = None

# Cumulative token counts for this run. Cache hits cost nothing and are counted
# separately so the printed totals stay honest about what was actually billed.
USAGE = {
    "calls": 0,
    "cache_hits": 0,
    "api_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


class LLMError(RuntimeError):
    """Raised when a call cannot be completed or its response cannot be parsed."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = anthropic.Anthropic()
    return _client


def cache_key(system: str, user: str, model: str) -> str:
    """SPEC.md section 7: cache on sha256(system + user + model)."""
    return hashlib.sha256((system + user + model).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return LLM_CACHE_DIR / (key + ".json")


def _request_kwargs(model: str) -> dict[str, Any]:
    """Per-model request parameters. See the sampling note above."""
    kwargs: dict[str, Any] = {}
    if _SAMPLING_PARAMS_AVAILABLE:
        kwargs["temperature"] = 0
    if model in _THINKING_ON_BY_DEFAULT:
        kwargs["thinking"] = {"type": "disabled"}
    return kwargs


def _extract_text(message: Any) -> str:
    return "".join(b.text for b in message.content if b.type == "text").strip()


def parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating code fences and surrounding prose."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        return json.loads(candidate[start:end + 1])
    raise json.JSONDecodeError("no JSON object found in response", candidate, 0)


def _create(system: str, user: str, model: str, max_tokens: int) -> Any:
    """One API call, retried twice on rate limit / 5xx with backoff."""
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **_request_kwargs(model),
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            last_error = exc
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                raise
            last_error = exc
        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
            print("  [llm] " + type(last_error).__name__ + f"; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise LLMError(
        f"{model}: giving up after {MAX_RETRIES + 1} attempts ({last_error})"
    )


def _record_usage(message: Any) -> None:
    USAGE["api_calls"] += 1
    USAGE["input_tokens"] += message.usage.input_tokens
    USAGE["output_tokens"] += message.usage.output_tokens


def call(system: str, user: str, model: str, max_tokens: int = 1024) -> dict:
    """Call Claude and return the parsed JSON response. Cached on disk."""
    USAGE["calls"] += 1
    key = cache_key(system, user, model)
    path = _cache_path(key)

    if path.exists():
        USAGE["cache_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["parsed"]

    message = _create(system, user, model, max_tokens)
    text = _extract_text(message)
    _record_usage(message)

    try:
        parsed = parse_json(text)
    except json.JSONDecodeError:
        # One repair attempt, then give up (SPEC.md section 7).
        repair = _create(system, user + REPAIR_SUFFIX, model, max_tokens)
        text = _extract_text(repair)
        _record_usage(repair)
        try:
            parsed = parse_json(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{model}: response was not valid JSON after one repair retry: {text[:300]}"
            ) from exc

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "model": model,
        "system": system,
        "user": user,
        "raw_text": text,
        "parsed": parsed,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "called_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed


def usage_summary() -> str:
    return (
        "llm: {calls} calls ({cache_hits} cached, {api_calls} to the API), "
        "{input_tokens} input tokens, {output_tokens} output tokens"
    ).format(**USAGE)


@atexit.register
def _print_usage() -> None:
    if USAGE["calls"]:
        print(usage_summary())


if __name__ == "__main__":
    load_dotenv()
    cached = len(list(LLM_CACHE_DIR.glob("*.json"))) if LLM_CACHE_DIR.exists() else 0
    print("judge model      = " + MODEL_JUDGE)
    print("generation model = " + MODEL_GENERATION)
    print(f"cache dir        = {LLM_CACHE_DIR} ({cached} cached responses)")
    print("api key set      = " + str(bool(os.environ.get("ANTHROPIC_API_KEY"))))
