
import asyncio
import os
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
from fastapi import types

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
    image_bytes: bytes,
    mime_type: str,
    model_name: str = GEMINI_MODEL,
    temp: float = 0.1,
    request_id: Optional[str] = None,
    call_type: str = "main"
) -> tuple[str, dict]:
    """
    Call Gemini API with image and prompt.
    Returns: (response_text, usage_dict)
    Raises: RuntimeError with "RATE_LIMIT_EXCEEDED" if rate limited
    """
    try:
        start = perf_counter()
        
        # Create the content parts
        contents = [
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))
        ]
        
        # Generate content with new SDK
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temp,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),
                ]
            )
        )
        
        duration_ms = (perf_counter() - start) * 1000.0
        
        # Extract usage metadata
        usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
            }
        
        # Log the call
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
        error_msg = str(e).lower()
        
        # Check if it's a rate limit error
        is_rate_limit = ("429" in error_msg or "quota" in error_msg or 
                       "rate" in error_msg or "resource" in error_msg or
                       "exhausted" in error_msg)
        
        if is_rate_limit:
            raise RuntimeError("RATE_LIMIT_EXCEEDED") from e
        
        # Re-raise other errors as-is
        raise
