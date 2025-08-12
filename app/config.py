import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_EXTRACT: str = os.getenv("OPENAI_MODEL_EXTRACT", "gpt-4o-mini")
    OPENAI_MODEL_REASON: str = os.getenv("OPENAI_MODEL_REASON", "gpt-4o-mini")
    # Soft caps for cost control
    MAX_OUTPUT_TOKENS_EXTRACT: int = int(os.getenv("MAX_OUTPUT_TOKENS_EXTRACT", 600))
    MAX_OUTPUT_TOKENS_REASON: int = int(os.getenv("MAX_OUTPUT_TOKENS_REASON", 400))

    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "scan_invoice")

    # Optional: default prompt if DB empty
    DEFAULT_SYSTEM_PROMPT: str = (
        os.getenv("DEFAULT_SYSTEM_PROMPT")
        or "You are a Professional Receipt & Invoice Analyzer. Return STRICT JSON only."
    )

settings = Settings()