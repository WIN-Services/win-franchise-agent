import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"          # cheap + capable; ~$0.60/1M tokens
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_DB_PATH: str = "vector_store"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3

    # ── Token budget controls ─────────────────────────────────────────────
    MAX_OUTPUT_TOKENS: int = 512          # cap LLM response length
    MAX_CONTEXT_TOKENS: int = 1500        # max tokens used for retrieved context
    MAX_HISTORY_MESSAGES: int = 20        # sliding window per session (10 turns)

    # ── Langfuse ──────────────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
