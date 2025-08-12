# app/prompt_store.py  — file-backed prompts (no Mongo)
import os
from typing import Optional

PROMPT_FILE = os.getenv("PROMPT_FILE", "app/data/system_prompt.txt")
FALLBACK = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a Professional Receipt & Invoice Analyzer. Return STRICT JSON only."
)

async def get_active_prompt(scope: str = "global") -> str:
    # scope kept for API compatibility, ignored for file mode
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt or FALLBACK
    except Exception:
        return FALLBACK

async def set_active_prompt(system_prompt: str, scope: str = "global") -> int:
    os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(system_prompt.rstrip() + "\n")
    return 1  # simple version number
