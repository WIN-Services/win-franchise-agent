import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-5.1"          # cheap + capable; ~$0.60/1M tokens
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_DB_PATH: str = "vector_store"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    TOP_K_RESULTS: int = 12
    RERANK_MODEL: str = "BAAI/bge-reranker-large"
    TOP_K_RERANK: int = 6

    # ── Token budget controls ─────────────────────────────────────────────
    MAX_OUTPUT_TOKENS: int = 512          # cap LLM response length
    MAX_CONTEXT_TOKENS: int = 2500        # max tokens used for retrieved context
    MAX_HISTORY_MESSAGES: int = 20        # sliding window per session (10 turns)

    # ── Langfuse ──────────────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
