"""Config loader — single source of truth for all settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field

from rag.exceptions import ConfigError

# Project root is two levels up from this file (rag/config.py → local-rag/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class LLMConfig(BaseModel):
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    request_timeout: float = 300.0
    temperature: float = 0.1
    context_window: int = 32768


class EmbeddingConfig(BaseModel):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    embed_batch_size: int = 10


class ChromaConfig(BaseModel):
    persist_dir: str = "storage/chroma"
    default_collection: str = "documents"
    distance_metric: str = "cosine"


class StorageConfig(BaseModel):
    docstore_dir: str = "storage/docstore"
    ingestion_cache_dir: str = "storage/ingestion_cache"
    data_dir: str = "data"


class IngestionConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    supported_extensions: List[str] = [".pdf", ".txt", ".docx", ".md", ".csv"]
    recursive: bool = True
    num_workers: int = 2


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    similarity_top_k: int = 5
    bm25_top_k: int = 5
    fusion_mode: str = "reciprocal_rerank"
    num_queries: int = 1
    similarity_cutoff: float = 0.1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    use_rich_console: bool = True


class AuthConfig(BaseModel):
    api_key: str = ""


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    def abs(self, rel_path: str) -> Path:
        """Resolve a relative path against the project root."""
        return PROJECT_ROOT / rel_path


def load_config(path: str | Path = None) -> AppConfig:
    """Load config.yaml and return a validated AppConfig."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config.yaml not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)
    except Exception as exc:
        raise ConfigError(f"Failed to load config: {exc}") from exc


# Module-level singleton — import this everywhere
settings: AppConfig = load_config()
