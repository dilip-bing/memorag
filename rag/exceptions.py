"""Custom exception hierarchy for the RAG pipeline."""


class RAGException(Exception):
    """Base class for all pipeline errors."""


class ConfigError(RAGException):
    """Bad or missing config values."""


class DocumentLoadError(RAGException):
    """File not found, unsupported format, or parse failure."""


class EmbeddingError(RAGException):
    """Ollama embedding service unreachable or returns error."""


class LLMError(RAGException):
    """Ollama LLM service unreachable or returns error."""


class StoreError(RAGException):
    """ChromaDB read/write failure."""


class RetrievalError(RAGException):
    """Retriever construction or query failure."""


class IngestionError(RAGException):
    """Wraps DocumentLoadError or EmbeddingError during batch ingest."""
