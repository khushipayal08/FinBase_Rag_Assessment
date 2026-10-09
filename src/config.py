"""Central configuration loaded from environment. No secrets in code."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- paths
    data_dir: Path = Field(default=Path("./data"))
    chroma_dir: Path = Field(default=Path("./data/chroma"))
    processed_dir: Path = Field(default=Path("./data/processed"))
    raw_pdf: Path = Field(default=Path("./data/raw/sample_1.pdf"))

    # ---- models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_model: str = "gemini-flash-latest"
    gemini_api_key: str | None = None

    # ---- retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    confidence_threshold: float = 0.35
    rrf_k: int = 60

    # ---- runtime
    log_level: str = "INFO"
    request_timeout_s: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()