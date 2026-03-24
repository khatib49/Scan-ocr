import asyncio
import os
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv

from google import genai
from google.genai import types

from utils.logger import append_llm_call

# Load environment variables
try:
    load_dotenv()
except Exception:
    pass

PROMPT_PATH = os.getenv("PROMPT_PATH", "data/prompt.txt")
with open(PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in environment or .env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Configure Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


async def call_gemini_with_image(
    prompt: str,
    mime_type: str,
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
    model_name: str = GEMINI_MODEL,
    temp: float = 0.1,
    request_id: Optional[str] = None,
    call_type: str = "main",
) -> tuple[str, dict]:
    """
    Call Gemini API with image and prompt.

    Accepts either:
      - image_bytes: raw bytes (used in /analyze where image is already in memory)
      - image_url:   Azure Blob / any public URL — Gemini fetches it directly,
                     no download happens on our server side

    Returns: (response_text, usage_dict)
    Raises:
      - ValueError if neither image_bytes nor image_url is provided
      - RuntimeError("RATE_LIMIT_EXCEEDED") if Gemini returns 429
      - Other exceptions re-raised as-is

    Retries up to 3 times on 503 UNAVAILABLE with exponential backoff (2s, 5s, 10s).
    """
    if image_bytes is None and image_url is None:
        raise ValueError("Either image_bytes or image_url must be provided.")

    # ── Build image part ───────────────────────────────────────────
    if image_bytes is not None:
        # Raw bytes — used in /analyze (image already in memory)
        image_part = types.Part(
            inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
        )
    else:
        # URL — Gemini fetches it directly, zero download overhead on our side
        print(f"[gemini] using image URL for {call_type} call")
        image_part = types.Part(
            file_data=types.FileData(mime_type=mime_type, file_uri=image_url)
        )

    contents = [types.Part(text=prompt), image_part]

    config = types.GenerateContentConfig(
        temperature=temp,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ],
    )

    # ── Retry loop ─────────────────────────────────────────────────
    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 5, 10]  # seconds between retries on 503
    last_error: Exception = None

    for attempt in range(MAX_RETRIES):
        try:
            start = perf_counter()

            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            )

            duration_ms = (perf_counter() - start) * 1000.0

            # Log finish reason for debugging truncation
            finish_reason = None
            if response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
            if finish_reason and str(finish_reason) not in (
                "STOP",
                "1",
                "FinishReason.STOP",
            ):
                print(f"[gemini] WARNING: {call_type} finish_reason={finish_reason}")

            # Extract usage metadata
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(
                        response.usage_metadata, "total_token_count", 0
                    ),
                }

            # Log the call
            if request_id:
                await append_llm_call(
                    request_id=request_id,
                    call_type=call_type,
                    model=model_name,
                    duration_ms=duration_ms,
                    usage=usage,
                )

            return response.text, usage

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # 429 rate limit — raise immediately, main.py handles it
            is_rate_limit = (
                "429" in error_msg
                or "quota" in error_msg
                or "rate" in error_msg
                or "resource" in error_msg
                or "exhausted" in error_msg
            )
            if is_rate_limit:
                raise RuntimeError("RATE_LIMIT_EXCEEDED") from e

            # 503 UNAVAILABLE — Gemini temporary overload, retry with backoff
            is_unavailable = "503" in error_msg or "unavailable" in error_msg
            if is_unavailable and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(
                    f"[gemini] 503 UNAVAILABLE on attempt {attempt + 1}/{MAX_RETRIES} "
                    f"({call_type}) — retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                continue

            # Any other error — raise immediately
            raise

    # All retries exhausted
    raise last_error
