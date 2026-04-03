"""
Local RAG API Server
- Built on the rag/ package (ChromaDB + LlamaIndex + Ollama)
- Accessible from any device on the network
- OpenAPI docs at /docs
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Security, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from rag.config import settings
from rag.logging_setup import setup_logging
from rag.store.chroma_store import ChromaManager
from rag.store.index_manager import IndexManager
from rag.ingest.pipeline import RAGIngestionPipeline
from rag.query.engine import RAGQueryEngine
from rag.models.requests import QueryRequest, IngestRequest, MemoryExtractRequest, GlobalMemoryUpdateRequest
from rag.models.responses import (
    QueryResponse,
    IngestResponse,
    CollectionStatsResponse,
    HealthResponse,
    SourceNodeModel,
    TaskSubmittedResponse,
    TaskStatusResponse,
)
import auth  # Our new auth module

setup_logging(settings)
logger = logging.getLogger("rag.api")


# ── API Key auth ───────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(_api_key_header)):
    configured = settings.auth.api_key
    if configured and key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Google Auth ─────────────────────────────────────────────────────────────

class GoogleAuthRequest(BaseModel):
    token: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    picture: str


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Extract and verify user from Authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization[7:]  # Remove "Bearer " prefix
    user_info = auth.verify_google_token(token)
    if not user_info:
        return None
    
    return auth.get_user_by_id(user_info['id'])


def require_auth(user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency that requires authentication"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# Protected router — all routes require API key only
# Google auth is a UI gate on the frontend; API key secures the backend
protected = APIRouter(dependencies=[Depends(verify_api_key)])


# ── App state container ────────────────────────────────────────────────────

class AppState:
    chroma: ChromaManager
    index_manager: IndexManager
    pipeline: RAGIngestionPipeline
    engine: RAGQueryEngine


state = AppState()

# ── Task store for async queries ──────────────────────────────────────────
tasks: Dict[str, Dict[str, Any]] = {}


# ── Lifespan (replaces deprecated on_event) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components once on startup, clean up on shutdown."""
    logger.info("Starting RAG API server...")

    state.chroma = ChromaManager(settings)
    state.index_manager = IndexManager(settings, state.chroma)
    state.pipeline = RAGIngestionPipeline(settings, state.chroma)
    state.engine = RAGQueryEngine(settings, state.index_manager)

    # Auto-ingest existing documents in data/ on startup
    data_dir = settings.abs(settings.storage.data_dir)
    if any(data_dir.iterdir()) if data_dir.exists() else False:
        logger.info("Auto-ingesting existing documents on startup...")
        result = state.pipeline.ingest_directory(data_dir)
        state.engine.invalidate_cache(settings.chroma.default_collection)
        logger.info(
            f"Startup ingest: {result.files_processed} files, "
            f"{result.nodes_created} nodes, {result.files_skipped} skipped"
        )

    logger.info("RAG API server ready.")
    yield
    logger.info("Shutting down RAG API server.")


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Local RAG API",
    description=(
        "Query your local documents using a fully local RAG pipeline.\n\n"
        "**Stack:** Ollama + LlamaIndex + ChromaDB + Hybrid Search (BM25 + Vector)\n\n"
        "**Models:** configurable via `config.yaml` — no restart needed for model swap.\n\n"
        "**Auth:** pass your API key via the `X-API-Key` header on all protected endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System endpoints ───────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Local RAG API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the API, Ollama, and ChromaDB are all reachable."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.llm.base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    chroma_ok = state.chroma.health_check()

    return HealthResponse(
        status="ok" if (ollama_ok and chroma_ok) else "degraded",
        ollama_reachable=ollama_ok,
        chroma_reachable=chroma_ok,
        version="1.0.0",
        active_llm=settings.llm.model,
        active_embedding=settings.embedding.model,
    )


@app.post("/auth/google", tags=["Auth"])
async def google_auth(request: GoogleAuthRequest):
    """
    Verify a Google ID token and register/update the user in the local DB.
    Called by the frontend after Google Sign-In.
    No API key required — this is the initial handshake.
    """
    user_info = auth.verify_google_token(request.token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    user = auth.get_or_create_user(user_info)
    logger.info(f"User authenticated: {user['email']}")

    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        picture=user.get("picture", ""),
    )


@app.get("/auth/me", tags=["Auth"])
async def get_me(authorization: Optional[str] = Header(None)):
    """Return the current user from a Google Bearer token (no API key needed)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token provided")
    token = authorization[7:]
    user_info = auth.verify_google_token(token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = auth.get_user_by_id(user_info["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found — please sign in first")
    return UserResponse(**{k: user[k] for k in ["id", "email", "name", "picture"]})


@protected.get("/config", tags=["System"])
async def get_config():
    """Return the active configuration (read from config.yaml)."""
    return {
        "llm_model": settings.llm.model,
        "embedding_model": settings.embedding.model,
        "retrieval_mode": settings.retrieval.mode,
        "chunk_size": settings.ingestion.chunk_size,
        "chunk_overlap": settings.ingestion.chunk_overlap,
        "similarity_top_k": settings.retrieval.similarity_top_k,
        "default_collection": settings.chroma.default_collection,
    }


# ── Query endpoints (async task-based) ─────────────────────────────────────

@protected.post("/query", response_model=TaskSubmittedResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Submit a RAG query. Returns a task_id immediately.
    Poll GET /query/{task_id} for the result.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None, "error": None}

    async def _run():
        try:
            result = await state.engine.aquery(
                question=request.question,
                collection=request.collection,
                top_k=request.top_k,
                thinking=request.thinking,
                model=request.model,
            )
            tasks[task_id]["result"] = QueryResponse(
                question=result.question,
                answer=result.answer,
                sources=[
                    SourceNodeModel(text=s.text, score=s.score, metadata=s.metadata)
                    for s in result.sources
                ],
                collection=result.collection,
                elapsed_seconds=result.elapsed_seconds,
            )
            tasks[task_id]["status"] = "completed"
        except Exception as exc:
            logger.error(f"Query error: {exc}", exc_info=True)
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(exc)

    asyncio.create_task(_run())
    return TaskSubmittedResponse(task_id=task_id)


@protected.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest):
    """
    Stream a RAG query with real-time status updates.
    Returns Server-Sent Events (SSE) stream.
    
    Event types:
    - status: {"type": "status", "status": "Thinking"|"Fast thinking", "phase": "retrieving"|"generating"}
    - answer: {"type": "answer", "content": "..."}
    - sources: {"type": "sources", "sources": [...]}
    - done: {"type": "done", "elapsed_seconds": 123.45, "collection": "documents"}
    - error: {"type": "error", "error": "error message"}
    """
    import json

    async def event_generator():
        try:
            async for chunk in state.engine.aquery_stream(
                question=request.question,
                collection=request.collection,
                top_k=request.top_k,
                thinking=request.thinking,
                model=request.model,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            logger.error(f"Stream error: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@protected.get("/query/{task_id}", response_model=TaskStatusResponse, tags=["RAG"])
async def query_status(task_id: str):
    """Poll this endpoint for query results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
    )


# ── Ingest endpoints ───────────────────────────────────────────────────────

@protected.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest(request: IngestRequest):
    """
    Ingest a file or directory path into the RAG pipeline.
    Skips unchanged files automatically (tracked by SHA-256 hash).
    Use `force=true` to re-embed everything.
    """
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

    try:
        if path.is_file():
            result = state.pipeline.ingest_file(
                path, collection=request.collection, force=request.force
            )
        else:
            result = state.pipeline.ingest_directory(
                path, collection=request.collection, force=request.force
            )
        state.engine.invalidate_cache(request.collection)

        return IngestResponse(
            status=result.status,
            files_processed=result.files_processed,
            nodes_created=result.nodes_created,
            files_skipped=result.files_skipped,
            errors=result.errors,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:
        logger.error(f"Ingest error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@protected.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    collection: str = "documents",
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a document (PDF, TXT, DOCX, MD, CSV).
    The file is saved to the data directory and ingested automatically.
    """
    allowed = set(settings.ingestion.supported_extensions)
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    data_dir = settings.abs(settings.storage.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / file.filename

    content = await file.read()
    save_path.write_bytes(content)
    logger.info(f"Uploaded: {file.filename} ({len(content)} bytes)")

    def _ingest():
        r = state.pipeline.ingest_file(save_path, collection=collection, force=True)
        state.engine.invalidate_cache(collection)
        logger.info(f"Background ingest complete: {r.nodes_created} nodes")

    if background_tasks:
        background_tasks.add_task(_ingest)
        ingest_status = "queued"
    else:
        _ingest()
        ingest_status = "done"

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size_bytes": len(content),
        "collection": collection,
        "ingest_status": ingest_status,
    }


@protected.get("/documents", tags=["Documents"])
async def list_documents():
    """List all files currently in the data directory."""
    data_dir = settings.abs(settings.storage.data_dir)
    if not data_dir.exists():
        return {"documents": [], "total_count": 0}

    files = []
    for f in data_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in settings.ingestion.supported_extensions:
            stat = f.stat()
            files.append({
                "filename": f.name,
                "relative_path": str(f.relative_to(data_dir)),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": f.suffix.lower(),
            })

    return {
        "documents": sorted(files, key=lambda x: x["modified_at"], reverse=True),
        "total_count": len(files),
    }


@protected.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str, collection: str = "documents"):
    """Delete a document from the data directory."""
    data_dir = settings.abs(settings.storage.data_dir)
    file_path = data_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    file_path.unlink()
    logger.info(f"Deleted: {filename}")
    return {"status": "deleted", "filename": filename}


# ── Collection endpoints ───────────────────────────────────────────────────

@protected.get("/collections", tags=["Collections"])
async def list_collections():
    """List all ChromaDB collections."""
    return {"collections": state.chroma.list_collections()}


@protected.get("/collections/{name}/stats", response_model=CollectionStatsResponse, tags=["Collections"])
async def collection_stats(name: str):
    """Get stats for a specific collection (document count, metadata)."""
    try:
        stats = state.chroma.collection_stats(name)
        return CollectionStatsResponse(
            name=stats["name"],
            document_count=stats["count"],
            metadata=stats["metadata"],
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@protected.delete("/collections/{name}", tags=["Collections"])
async def delete_collection(name: str):
    """Delete a ChromaDB collection and all its embeddings."""
    if name == settings.chroma.default_collection:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete the default collection '{name}'. Use a different name."
        )
    try:
        state.chroma.delete_collection(name)
        state.engine.invalidate_cache(name)
        return {"status": "deleted", "collection": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@protected.get("/models", tags=["System"])
async def list_models():
    """List all models available in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.llm.base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [
                {
                    "name": m["name"],
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in data.get("models", [])
            ]
            return {
                "models": models,
                "active_model": settings.llm.model,
            }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}")


# ── Memory endpoints ───────────────────────────────────────────────────────

@protected.post("/memory/extract", tags=["Memory"])
async def extract_memory(request: MemoryExtractRequest):
    """
    Use the local LLM to convert raw brain-dump text into structured memory cards.
    Returns a list of typed, tagged cards ready to display in the UI.
    """
    if not request.text.strip():
        return {"cards": []}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a memory extraction assistant. "
                "Extract structured facts from text and return ONLY a JSON array. "
                "No markdown fences, no explanation — raw JSON array only."
            ),
        },
        {
            "role": "user",
            "content": f"""Extract memory cards from this text.

Text: \"{request.text}\"

Return a JSON array where each object has:
- "type": one of "fact" | "preference" | "context" | "skill" | "goal"
- "content": clear, standalone statement (max 120 chars)
- "importance": "high" | "medium" | "low"
- "tags": array of 1-3 short lowercase tags

Type guide:
  fact        — who they are (name, job, location, background)
  preference  — how they like things (style, format, level of detail)
  context     — current project, situation, or challenge
  skill       — expertise, tools, languages they know
  goal        — what they're trying to achieve

Rules:
  - Only extract what is explicitly stated — NO hallucination
  - Make each card standalone and self-contained
  - Return [] if nothing useful is found

JSON array:""",
        },
    ]

    import re, json as _json

    try:
        async with httpx.AsyncClient(timeout=90) as client:   # 90s — model may be busy
            r = await client.post(
                f"{settings.llm.base_url}/api/chat",
                json={
                    "model": settings.llm.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,          # suppress thinking for speed
                    "options": {"temperature": 0.1, "num_predict": 1000},
                },
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=502, detail="LLM timed out — try again in a moment")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM unreachable: {exc}")

    # Strip <think>...</think> blocks that reasoning models may inject
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract the JSON array (tolerates extra prose before/after)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        logger.warning(f"Memory extract: no JSON array in response: {raw[:200]}")
        return {"cards": []}

    try:
        raw_cards = _json.loads(match.group())
    except _json.JSONDecodeError as exc:
        logger.warning(f"Memory extract: JSON parse failed: {exc} — raw: {raw[:200]}")
        return {"cards": []}

    valid_types = {'fact', 'preference', 'context', 'skill', 'goal'}
    valid_imp   = {'high', 'medium', 'low'}
    now = int(datetime.utcnow().timestamp() * 1000)
    cards = []
    for c in raw_cards:
        if not isinstance(c, dict) or not c.get('content'):
            continue
        cards.append({
            "id":         str(uuid.uuid4()),
            "type":       c.get("type") if c.get("type") in valid_types else "fact",
            "content":    str(c.get("content", ""))[:200].strip(),
            "importance": c.get("importance") if c.get("importance") in valid_imp else "medium",
            "tags":       [str(t).lower()[:30] for t in c.get("tags", []) if t][:5],
            "createdAt":  now,
        })

    return {"cards": cards}


@protected.get("/memory/global", tags=["Memory"])
async def get_global_memory(user_id: str):
    """Return a user's global (cross-chat) memory cards."""
    if not auth.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    cards = auth.get_global_memory(user_id)
    return {"cards": cards, "count": len(cards)}


@protected.put("/memory/global", tags=["Memory"])
async def update_global_memory(user_id: str, request: GlobalMemoryUpdateRequest):
    """Replace a user's global memory cards (full overwrite)."""
    if not auth.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    cards = [c.model_dump() for c in request.cards]
    ok = auth.update_global_memory(user_id, cards)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save memory")
    return {"status": "saved", "count": len(cards)}


app.include_router(protected)


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*55)
    print("  Local RAG API starting...")
    print("="*55)
    print(f"  Local:    http://localhost:8000")
    print(f"  Network:  http://{local_ip}:8000")
    print(f"  Docs:     http://localhost:8000/docs")
    print(f"  LLM:      {settings.llm.model}")
    print(f"  Embed:    {settings.embedding.model}")
    print(f"  Mode:     {settings.retrieval.mode}")
    print(f"  API Key:  {settings.auth.api_key[:8]}...")
    print("="*55 + "\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
