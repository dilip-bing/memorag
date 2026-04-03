"""API request models."""

from typing import Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest")
    collection: str = Field("documents", description="ChromaDB collection name")
    recursive: bool = Field(True, description="Recurse into subdirectories")
    force: bool = Field(False, description="Re-ingest even if file is unchanged")


class ChatMessage(BaseModel):
    """Single message from conversation history."""
    role: str   # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask")
    collection: str = Field("documents", description="Fallback ChromaDB collection")
    top_k: Optional[int] = Field(None, description="Override retrieval top_k")
    thinking: bool = Field(True, description="Enable thinking/reasoning mode")
    model: Optional[str] = Field(None, description="Override LLM model")
    # Session context
    chat_id: Optional[str] = Field(None, description="Current chat ID (for per-chat doc lookup)")
    user_id: Optional[str] = Field(None, description="Current user ID (for profile doc lookup)")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="Last N messages for context")


class CollectionRequest(BaseModel):
    name: str = Field(..., description="Collection name")


class MemoryExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw brain-dump text to extract memory from")


class MemoryCardModel(BaseModel):
    id: str
    type: str  # fact | preference | context | skill | goal
    content: str
    importance: str  # high | medium | low
    tags: list[str] = []
    createdAt: int


class GlobalMemoryUpdateRequest(BaseModel):
    cards: list[MemoryCardModel]
