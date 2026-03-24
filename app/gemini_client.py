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

# ── API Key Pool ───────────────────────────────────────────────────
# Load all available API keys from environment
# Primary key is required, backup keys are optional
def _load_api_keys() -> list[str]:
    keys = []
    # Primary key — required
    primary = os.getenv("GEMINI_API_KEY")
    if not primary:
        raise RuntimeError("Set GEMINI_API_KEY in environment or .env")
    keys.append(primary)
    # Backup keys — optional, add as many as you have
    for i in range(2, 10):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
        else:
            break  # stop at first missing key
    print(f"[gemini] Loaded {len(keys)} API key(s)")
    return keys

API_KEYS = _load_api_keys()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Client Pool ────────────────────────────────────────────────────
# One client per key, indexed same as API_KEYS
_clients = [genai.Client(api_key=key) for key in API_KEYS]

# Current active key index (shared across calls)
_current_key_index = 0


def _get_client() -> tuple[genai.Client, int]:
    """Returns the current active client and its index."""
    return _clients[_current_key_index], _current_key_index


def _rotate_key() -> tuple[genai.Client, int]:
    """
    Rotates to the next available API key.
    Returns (new_client, new_index) or raises RuntimeError if no more keys.
    """
    global _current_key_index
    next_index = _current_key_index + 1
    if next_index >= len(_clients):
        raise RuntimeError(
            f"RATE_LIMIT_EXCEEDED — all {len(API_KEYS)} API key(s) exhausted"
        )
    _current_key_index = next_index
    print(f"[gemini] Rotated to API key #{next_index + 1} of {len(API_KEYS)}")
    return _clients[_current_key_index], _current_key_index


async def call_gemini_with_image(
    prompt: str,
    mime_type: str = "image/jpeg",
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
    model_name: str = GEMINI_MODEL,
    temp: float = 0.1,
    request_id: Optional[str] = None,
    call_type: str = "main"
) -> tuple[str, dict]:
    """
    Call Gemini API with image and prompt.

    Accepts either:
      - image_bytes: raw bytes (used in /analyze where image is already in memory)
      - image_url:   Azure Blob / any public URL — Gemini fetches it directly

    Retry behavior:
      - 503 UNAVAILABLE: retries 3x with short backoff (2s, 5s, 10s)
      - 429 RATE LIMIT:  retries 3x with long backoff (30s, 60s, 90s)
                         if still failing → rotates to next API key and retries
                         if all keys exhausted → raises RATE_LIMIT_EXCEEDED
      - Other errors:    raises immediately, no retry

    Returns: (response_text, usage_dict)
    """
    if image_bytes is None and image_url is None:
        raise ValueError("Either image_bytes or image_url must be provided.")

    # ── Build image part ───────────────────────────────────────────
    if image_bytes is not None:
        image_part = types.Part(
            inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
        )
    else:
        print(f"[gemini] using image URL for {call_type} call")
        image_part = types.Part(
            file_data=types.FileData(mime_type=mime_type, file_uri=image_url)
        )

    contents = [
        types.Part(text=prompt),
        image_part
    ]

    config = types.GenerateContentConfig(
        temperature=temp,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
    )

    # ── Retry + Key Rotation Loop ──────────────────────────────────
    MAX_RETRIES_PER_KEY = 3
    RETRY_DELAYS_503    = [2,  5,  10]   # short backoff — temporary overload
    RETRY_DELAYS_429    = [30, 60, 90]   # long backoff  — quota exhausted

    # Outer loop: iterate over available API keys
    while True:
        client, key_idx = _get_client()
        last_error: Exception = None

        # Inner loop: retry with current key
        for attempt in range(MAX_RETRIES_PER_KEY):
            try:
                start = perf_counter()

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config
                )

                duration_ms = (perf_counter() - start) * 1000.0

                # Extract usage metadata
                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "prompt_tokens":     getattr(response.usage_metadata, "prompt_token_count",     0),
                        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "total_tokens":      getattr(response.usage_metadata, "total_token_count",      0),
                    }

                if request_id:
                    await append_llm_call(
                        request_id=request_id,
                        call_type=call_type,
                        model=model_name,
                        duration_ms=duration_ms,
                        usage=usage
                    )

                return response.text, usage

            except Exception as e:
                last_error = e
                error_msg  = str(e).lower()

                # ── 429 Rate Limit ────────────────────────────────
                is_rate_limit = (
                    "429" in error_msg or "quota" in error_msg or
                    "rate" in error_msg or "resource" in error_msg or
                    "exhausted" in error_msg
                )
                if is_rate_limit:
                    if attempt < MAX_RETRIES_PER_KEY - 1:
                        delay = RETRY_DELAYS_429[attempt]
                        print(f"[gemini] 429 RATE LIMIT on key #{key_idx + 1}, "
                              f"attempt {attempt + 1}/{MAX_RETRIES_PER_KEY} "
                              f"({call_type}) — retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # All retries on this key failed — try next key
                        print(f"[gemini] 429 RATE LIMIT — key #{key_idx + 1} exhausted all retries, rotating key...")
                        break  # exit inner loop → rotate key

                # ── 503 Unavailable ───────────────────────────────
                is_unavailable = "503" in error_msg or "unavailable" in error_msg
                if is_unavailable and attempt < MAX_RETRIES_PER_KEY - 1:
                    delay = RETRY_DELAYS_503[attempt]
                    print(f"[gemini] 503 UNAVAILABLE on key #{key_idx + 1}, "
                          f"attempt {attempt + 1}/{MAX_RETRIES_PER_KEY} "
                          f"({call_type}) — retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # ── Any other error ───────────────────────────────
                raise

        # Inner loop done — check if we exhausted due to rate limit
        # Try to rotate to next key
        try:
            _rotate_key()
            # Successfully rotated — continue outer loop with new key
            print(f"[gemini] Retrying with new API key...")
        except RuntimeError:
            # No more keys available — give up
            raise RuntimeError("RATE_LIMIT_EXCEEDED") from last_error