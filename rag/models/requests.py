"""API request models."""

from typing import Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest")
    collection: str = Field("documents", description="ChromaDB collection name")
    recursive: bool = Field(True, description="Recurse into subdirectories")
    force: bool = Field(False, description="Re-ingest even if file is unchanged")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask")
    collection: str = Field("documents", description="ChromaDB collection to query")
    top_k: Optional[int] = Field(None, description="Override retrieval top_k")
    thinking: bool = Field(True, description="Enable thinking/reasoning (slower but better). Set false for fast answers.")
    model: Optional[str] = Field(None, description="Override LLM model (e.g. 'llama3.1:latest')")


class CollectionRequest(BaseModel):
    name: str = Field(..., description="Collection name")
