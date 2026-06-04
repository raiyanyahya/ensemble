from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

# Status codes worth retrying: rate limits and transient upstream failures.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_TIMEOUT = 120.0


class ModelError(RuntimeError):
    """Raised when a provider call fails non-transiently or out of retries."""


@dataclass
class Usage:
    """Token accounting for one model call, normalized across providers.

    ``input_tokens`` is the total prompt tokens; ``cached_tokens`` is the subset
    served (read) from the provider's prompt cache, billed at a discount;
    ``cache_creation_tokens`` is the subset written *into* the cache, which some
    providers (Anthropic) bill at a premium over the base input rate.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class ModelResult:
    text: str
    usage: Usage


async def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """POST JSON with exponential backoff + jitter on transient failures."""
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                    await asyncio.sleep(_retry_delay(resp, attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                last_exc = ModelError(f"HTTP {e.response.status_code}: {body}")
                if e.response.status_code not in RETRYABLE_STATUS or attempt >= max_attempts:
                    raise last_exc from e
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = ModelError(f"network error: {e}")
                if attempt >= max_attempts:
                    raise last_exc from e
            await asyncio.sleep(_backoff(attempt))
    raise last_exc or ModelError("request failed")


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return _backoff(attempt)


def _backoff(attempt: int) -> float:
    return min(2.0 ** attempt + random.uniform(0, 0.5), 30.0)


# --- Provider call implementations (return text + normalized usage) ----------


async def call_openai(system_prompt: str, user_prompt: str, model_id: str) -> ModelResult:
    # OpenAI caches stable prefixes automatically for prompts >~1024 tokens.
    data = await _post_json(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {_key('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        payload={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    usage = Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        cached_tokens=cached,
    )
    return ModelResult(data["choices"][0]["message"]["content"], usage)


async def call_anthropic(system_prompt: str, user_prompt: str, model_id: str) -> ModelResult:
    # Mark the (stable) system prompt as a cache breakpoint so repeated rounds
    # reuse it at the cache-read rate.
    data = await _post_json(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _key("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        payload={
            "model": model_id,
            "system": [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )
    u = data.get("usage") or {}
    cache_read = u.get("cache_read_input_tokens", 0)
    cache_creation = u.get("cache_creation_input_tokens", 0)
    total_in = u.get("input_tokens", 0) + cache_read + cache_creation
    usage = Usage(
        input_tokens=total_in,
        output_tokens=u.get("output_tokens", 0),
        cached_tokens=cache_read,
        cache_creation_tokens=cache_creation,
    )
    return ModelResult(data["content"][0]["text"], usage)


async def call_deepseek(system_prompt: str, user_prompt: str, model_id: str) -> ModelResult:
    # DeepSeek does context caching on disk automatically.
    data = await _post_json(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {_key('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        },
        payload={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )
    u = data.get("usage") or {}
    usage = Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        cached_tokens=u.get("prompt_cache_hit_tokens", 0),
    )
    return ModelResult(data["choices"][0]["message"]["content"], usage)


def _key(env_var: str) -> str:
    # Read at call time, not import time, so env set by the CLI/tests is seen.
    key = os.environ.get(env_var)
    if not key:
        raise ModelError(f"{env_var} is not set or empty — cannot call provider")
    return key


# --- Provider registry (single source of truth) -----------------------------


@dataclass(frozen=True)
class Provider:
    key: str
    display: str
    env_var: str
    default_model: str
    price_in: float  # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens
    call: Callable[[str, str, str], Awaitable[ModelResult]]
    cache_read_discount: float = 0.5  # fraction of input price for cache *reads*
    cache_write_multiplier: float = 1.0  # multiple of input price for cache *writes*


PROVIDERS: dict[str, Provider] = {
    "gpt4o": Provider(
        key="gpt4o",
        display="GPT-4o Mini (OpenAI)",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        price_in=0.15,
        price_out=0.60,
        cache_read_discount=0.5,
        call=call_openai,
    ),
    "claude": Provider(
        key="claude",
        display="Claude Haiku 4.5 (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-haiku-4-5-20251001",
        price_in=1.00,
        price_out=5.00,
        cache_read_discount=0.1,
        cache_write_multiplier=1.25,
        call=call_anthropic,
    ),
    # A strong single-model baseline for the eval. Not in DEFAULT_ORDER, so it
    # never joins the default ensemble or the CLI provider table — it is only
    # used when explicitly requested (e.g. `ensemble-eval --baseline sonnet`).
    # Registering it as a real provider means usage_cost() can price it.
    "sonnet": Provider(
        key="sonnet",
        display="Claude Sonnet 4.6 (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        price_in=3.00,
        price_out=15.00,
        cache_read_discount=0.1,
        cache_write_multiplier=1.25,
        call=call_anthropic,
    ),
    "deepseek": Provider(
        key="deepseek",
        display="DeepSeek Chat",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        price_in=0.27,
        price_out=1.10,
        cache_read_discount=0.1,
        call=call_deepseek,
    ),
}


def get_api_key(name: str) -> str | None:
    p = PROVIDERS.get(name)
    if not p:
        return None
    return os.environ.get(p.env_var) or None


def provider_name(name: str) -> str:
    p = PROVIDERS.get(name)
    return p.display if p else name


def default_model_id(name: str) -> str:
    p = PROVIDERS.get(name)
    return p.default_model if p else name


def usage_cost(provider_key: str, model_id: str, usage: Usage) -> float | None:
    """Estimated USD cost for a call, or None if pricing is unknown.

    Pricing is only known for each provider's default model (custom / OpenRouter
    model ids return None, so the debate still tracks tokens without a $ figure).
    """
    p = PROVIDERS.get(provider_key)
    if not p or model_id != p.default_model:
        return None
    non_cached = max(usage.input_tokens - usage.cached_tokens - usage.cache_creation_tokens, 0)
    cost = (
        non_cached * p.price_in
        + usage.cached_tokens * p.price_in * p.cache_read_discount
        + usage.cache_creation_tokens * p.price_in * p.cache_write_multiplier
        + usage.output_tokens * p.price_out
    ) / 1_000_000
    return cost


