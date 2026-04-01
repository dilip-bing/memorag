"""API response models."""

from typing import List, Dict, Any
from pydantic import BaseModel


class SourceNodeModel(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceNodeModel]
    collection: str
    elapsed_seconds: float


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    nodes_created: int
    files_skipped: int
    errors: List[str]
    elapsed_seconds: float


class CollectionStatsResponse(BaseModel):
    name: str
    document_count: int
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_reachable: bool
    version: str
    active_llm: str
    active_embedding: str
