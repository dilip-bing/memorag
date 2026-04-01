"""
Local RAG API Server
- Built on the rag/ package (ChromaDB + LlamaIndex + Ollama)
- Accessible from any device on the network
- OpenAPI docs at /docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.security.api_key import APIKeyHeader

from rag.config import settings
from rag.logging_setup import setup_logging
from rag.store.chroma_store import ChromaManager
from rag.store.index_manager import IndexManager
from rag.ingest.pipeline import RAGIngestionPipeline
from rag.query.engine import RAGQueryEngine
from rag.models.requests import QueryRequest, IngestRequest
from rag.models.responses import (
    QueryResponse,
    IngestResponse,
    CollectionStatsResponse,
    HealthResponse,
    SourceNodeModel,
)

setup_logging(settings)
logger = logging.getLogger("rag.api")


# ── API Key auth ───────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(_api_key_header)):
    configured = settings.auth.api_key
    if configured and key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Protected router — all routes added to this require a valid API key
protected = APIRouter(dependencies=[Depends(verify_api_key)])


# ── App state container ────────────────────────────────────────────────────

class AppState:
    chroma: ChromaManager
    index_manager: IndexManager
    pipeline: RAGIngestionPipeline
    engine: RAGQueryEngine


state = AppState()


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


# ── Query endpoint ─────────────────────────────────────────────────────────

@protected.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Ask a question — the RAG pipeline retrieves relevant context
    from your documents and generates a grounded answer.
    """
    try:
        result = await state.engine.aquery(
            question=request.question,
            collection=request.collection,
            top_k=request.top_k,
            thinking=request.thinking,
        )
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                SourceNodeModel(
                    text=s.text,
                    score=s.score,
                    metadata=s.metadata,
                )
                for s in result.sources
            ],
            collection=result.collection,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:
        logger.error(f"Query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


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
