"""
Configuration management for RAG engine.
Supports environment variables and defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class RAGConfig(BaseSettings):
    """RAG configuration with environment variable support."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    CHROMA_DB_PATH: Path = STORAGE_DIR / "chroma"
    DOCSTORE_PATH: Path = STORAGE_DIR / "docstore"
    INGESTION_CACHE_PATH: Path = STORAGE_DIR / "ingestion_cache"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # LLM & Embedding Models
    LLM_MODEL: str = "qwen2.5:7b"  # Can be overridden via env var
    EMBED_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # RAG Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5
    TEMPERATURE: float = 0.7

    # Processing
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 32

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)
        self.INGESTION_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global config instance
config = RAGConfig()
