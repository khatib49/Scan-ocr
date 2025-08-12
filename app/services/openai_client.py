from openai import OpenAI
from typing import Dict, Any
from ..config import settings

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client

async def gpt_extract(image_b64: str, system_prompt: str) -> Dict[str, Any]:
    client = get_client()
    # Single multimodal extraction call (image + JSON instructions)
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL_EXTRACT,
        max_tokens=settings.MAX_OUTPUT_TOKENS_EXTRACT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the required fields. Return ONLY valid JSON."},
                    {"type": "input_image", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
    )
    choice = resp.choices[0]
    return {
        "text": choice.message.content,
        "usage": resp.usage.model_dump() if hasattr(resp, "usage") else {},
    }

async def gpt_reason(extracted_json: dict, validation: dict, system_prompt: str) -> Dict[str, Any]:
    client = get_client()
    # Small text-only call for narrative/justification (cheaper)
    user_text = (
        "You are given extracted receipt fields (JSON) and code-side validation results. "
        "Write a concise explanation of the fraudScore/confidence, and state key mismatches if any. "
        "Return a JSON with keys: reason (string)."
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL_REASON,
        max_tokens=settings.MAX_OUTPUT_TOKENS_REASON,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "user", "content": f"EXTRACTED=```json\n{extracted_json}\n```"},
            {"role": "user", "content": f"VALIDATION=```json\n{validation}\n```"},
        ],
    )
    choice = resp.choices[0]
    return {
        "text": choice.message.content,
        "usage": resp.usage.model_dump() if hasattr(resp, "usage") else {},
    }