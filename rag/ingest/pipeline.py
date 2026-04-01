"""RAG ingestion pipeline — load, chunk, embed, store."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from rag.config import AppConfig
from rag.exceptions import IngestionError
from rag.ingest.loader import DocumentLoader
from rag.ingest.tracker import IngestionTracker
from rag.store.chroma_store import ChromaManager

logger = logging.getLogger("rag.ingest.pipeline")


@dataclass
class IngestResult:
    files_processed: int = 0
    nodes_created: int = 0
    files_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def status(self) -> str:
        return "error" if self.errors and self.files_processed == 0 else "ok"


class RAGIngestionPipeline:
    def __init__(self, config: AppConfig, chroma_manager: ChromaManager):
        self.config = config
        self.chroma = chroma_manager
        self.loader = DocumentLoader(config.ingestion)
        self.tracker = IngestionTracker(
            config.abs(config.storage.ingestion_cache_dir) / "tracker.json"
        )
        self.splitter = SentenceSplitter(
            chunk_size=config.ingestion.chunk_size,
            chunk_overlap=config.ingestion.chunk_overlap,
        )
        self.embed_model = OllamaEmbedding(
            model_name=config.embedding.model,
            base_url=config.embedding.base_url,
            embed_batch_size=config.embedding.embed_batch_size,
        )
        Settings.embed_model = self.embed_model
        logger.info(
            f"Pipeline ready - LLM embed: {config.embedding.model}, "
            f"chunk: {config.ingestion.chunk_size}/{config.ingestion.chunk_overlap}"
        )

    def ingest_file(
        self, path: Path, collection: str = "documents", force: bool = False
    ) -> IngestResult:
        """Ingest a single file."""
        path = Path(path)
        start = time.time()
        result = IngestResult()

        if not force and not self.tracker.is_changed(path):
            logger.info(f"Skipping unchanged file: {path.name}")
            result.files_skipped = 1
            result.elapsed_seconds = time.time() - start
            return result

        try:
            docs = self.loader.load_file(path)
            nodes = self.splitter.get_nodes_from_documents(docs)
            vector_store = self.chroma.get_vector_store(collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex(nodes=nodes, storage_context=storage_context)
            self.tracker.mark_ingested(path, collection)
            result.files_processed = 1
            result.nodes_created = len(nodes)
            logger.info(f"Ingested {path.name} -> {len(nodes)} nodes")
        except Exception as exc:
            msg = f"Failed to ingest {path.name}: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        result.elapsed_seconds = time.time() - start
        return result

    def ingest_directory(
        self,
        directory: Path,
        collection: str = "documents",
        force: bool = False,
    ) -> IngestResult:
        """Ingest all supported files from a directory."""
        directory = Path(directory)
        start = time.time()
        total = IngestResult()

        supported = set(self.config.ingestion.supported_extensions)
        pattern = "**/*" if self.config.ingestion.recursive else "*"
        files = [
            f
            for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported
        ]

        if not files:
            logger.warning(f"No supported files found in {directory}")
            total.elapsed_seconds = time.time() - start
            return total

        logger.info(f"Found {len(files)} file(s) to process in {directory}")

        for f in files:
            r = self.ingest_file(f, collection=collection, force=force)
            total.files_processed += r.files_processed
            total.nodes_created += r.nodes_created
            total.files_skipped += r.files_skipped
            total.errors.extend(r.errors)

        total.elapsed_seconds = time.time() - start
        logger.info(
            f"Done — {total.files_processed} ingested, "
            f"{total.files_skipped} skipped, "
            f"{len(total.errors)} errors, "
            f"{total.nodes_created} nodes, "
            f"{total.elapsed_seconds:.1f}s"
        )
        return total
