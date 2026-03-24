from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # --- Groq ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"          # fast open-source LLM via Groq

    # --- Embeddings ---
    # Using sentence-transformers locally (free, no extra API key)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 384-dim, fast & accurate

    # --- Chunking ---
    CHUNK_SIZE: int = 512          # tokens / chars per chunk
    CHUNK_OVERLAP: int = 64        # overlap between adjacent chunks

    # --- Retrieval ---
    TOP_K_CHUNKS: int = 5          # chunks injected into each prompt

    # --- Conversation window ---
    WINDOW_MEMORY_SIZE: int = 10   # last N interaction pairs kept

    # --- Cost estimation (Groq llama3-8b pricing, USD per 1M tokens) ---
    INPUT_COST_PER_1M: float = 0.05
    OUTPUT_COST_PER_1M: float = 0.08

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
