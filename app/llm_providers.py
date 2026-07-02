# app/llm_providers.py
"""
Provider-agnostic LLM layer.

Wraps each vendor SDK behind one interface so the pipeline can switch
models/providers via env vars — no code changes needed.

Env vars:
  EXTRACTOR_PROVIDER   gemini | claude              (default: gemini)
  EXTRACTOR_MODEL      model name for the extractor (default: gemini-3-flash-preview)
  ARBITRATION_ENABLED  true | false                 (default: true if ANTHROPIC_API_KEY set)
  ARBITRATOR_PROVIDER  claude | gemini              (default: claude)
  ARBITRATOR_MODEL     model for arbitration        (default: claude-sonnet-5)
  ANTHROPIC_API_KEY    required for the Claude provider
"""

import base64
import os
from time import perf_counter
from typing import Optional

import aiohttp
from dotenv import load_dotenv

from app.gemini_client import call_gemini_with_image, GEMINI_MODEL
from utils.logger import append_llm_call

try:
    load_dotenv()
except Exception:
    pass


# ────────────────────────────────────────────────────────────────
# Base interface
# ────────────────────────────────────────────────────────────────

class BaseProvider:
    """One method: generate(prompt, image) -> (json_text, usage_dict)."""

    name: str = "base"
    model: str = ""

    async def generate(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        mime_type: str = "image/jpeg",
        temp: float = 0.1,
        request_id: Optional[str] = None,
        call_type: str = "main",
        response_schema: Optional[dict] = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────
# Gemini (reuses existing retry + key-rotation logic)
# ────────────────────────────────────────────────────────────────

class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        self.model = model or GEMINI_MODEL

    async def generate(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        mime_type: str = "image/jpeg",
        temp: float = 0.1,
        request_id: Optional[str] = None,
        call_type: str = "main",
        response_schema: Optional[dict] = None,
    ) -> tuple[str, dict]:
        return await call_gemini_with_image(
            prompt=prompt,
            image_bytes=image_bytes,
            image_url=image_url,
            mime_type=mime_type,
            model_name=self.model,
            temp=temp,
            request_id=request_id,
            call_type=call_type,
            response_schema=response_schema,
        )


# ────────────────────────────────────────────────────────────────
# Claude (Anthropic)
# ────────────────────────────────────────────────────────────────

class ClaudeProvider(BaseProvider):
    name = "claude"

    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set — Claude provider unavailable")
        from anthropic import AsyncAnthropic  # imported lazily so gemini-only deployments don't need it

        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model or os.getenv("ARBITRATOR_MODEL", "claude-sonnet-5")

    @staticmethod
    async def _fetch_bytes(url: str, timeout_s: int = 15) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def generate(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        mime_type: str = "image/jpeg",
        temp: float = 0.1,
        request_id: Optional[str] = None,
        call_type: str = "main",
        response_schema: Optional[dict] = None,  # noqa: ARG002 — schema enforced via prompt for Claude
    ) -> tuple[str, dict]:
        if image_bytes is None and image_url is None:
            raise ValueError("Either image_bytes or image_url must be provided.")
        if image_bytes is None:
            image_bytes = await self._fetch_bytes(image_url)

        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        start = perf_counter()

        msg = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temp,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        duration_ms = (perf_counter() - start) * 1000.0

        text = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        usage = {
            "prompt_tokens": getattr(msg.usage, "input_tokens", 0),
            "completion_tokens": getattr(msg.usage, "output_tokens", 0),
            "total_tokens": getattr(msg.usage, "input_tokens", 0)
            + getattr(msg.usage, "output_tokens", 0),
        }

        if request_id:
            await append_llm_call(
                request_id=request_id,
                call_type=call_type,
                model=self.model,
                duration_ms=duration_ms,
                usage=usage,
            )

        return text, usage


# ────────────────────────────────────────────────────────────────
# Factories (cached singletons)
# ────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, BaseProvider] = {}


def _build(provider: str, model: Optional[str]) -> BaseProvider:
    key = f"{provider}:{model}"
    if key not in _PROVIDERS:
        if provider == "claude":
            _PROVIDERS[key] = ClaudeProvider(model)
        else:
            _PROVIDERS[key] = GeminiProvider(model)
    return _PROVIDERS[key]


def get_extractor() -> BaseProvider:
    provider = os.getenv("EXTRACTOR_PROVIDER", "gemini").strip().lower()
    model = os.getenv("EXTRACTOR_MODEL", "gemini-3-flash-preview").strip() or None
    return _build(provider, model)


def get_arbitrator() -> Optional[BaseProvider]:
    """Returns the arbitration provider, or None when arbitration is disabled."""
    enabled = os.getenv(
        "ARBITRATION_ENABLED",
        "true" if os.getenv("ANTHROPIC_API_KEY") else "false",
    ).strip().lower() in ("1", "true", "yes")
    if not enabled:
        return None
    provider = os.getenv("ARBITRATOR_PROVIDER", "claude").strip().lower()
    model = os.getenv("ARBITRATOR_MODEL", "claude-sonnet-5").strip() or None
    try:
        return _build(provider, model)
    except RuntimeError as e:
        print(f"[llm_providers] Arbitrator unavailable: {e}")
        return None
