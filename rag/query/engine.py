"""RAG query engine — retrieves context and generates answers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from rag.config import AppConfig
from rag.exceptions import LLMError, RetrievalError
from rag.store.index_manager import IndexManager
from rag.retrieve.hybrid_retriever import build_retriever

logger = logging.getLogger("rag.query.engine")


@dataclass
class SourceNode:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: List[SourceNode]
    collection: str
    elapsed_seconds: float


class RAGQueryEngine:
    def __init__(self, config: AppConfig, index_manager: IndexManager):
        self.config = config
        self.index_manager = index_manager
        self._engines: Dict[str, RetrieverQueryEngine] = {}

        # Two LLM instances — switched per request based on thinking flag
        self._llm_think = Ollama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            request_timeout=config.llm.request_timeout * 3,  # thinking needs more time
            temperature=config.llm.temperature,
            context_window=config.llm.context_window,
            thinking=True,
        )
        self._llm_fast = Ollama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            request_timeout=config.llm.request_timeout,
            temperature=config.llm.temperature,
            context_window=config.llm.context_window,
            thinking=False,
        )
        Settings.llm = self._llm_fast  # default to fast
        logger.info(f"Query engine ready - LLM: {config.llm.model}")

    def _get_llm(self, thinking: bool, model: Optional[str] = None) -> Ollama:
        """Return the right LLM instance. Creates a new one if model is overridden."""
        if model and model != self.config.llm.model:
            return Ollama(
                model=model,
                base_url=self.config.llm.base_url,
                request_timeout=self.config.llm.request_timeout * (3 if thinking else 1),
                temperature=self.config.llm.temperature,
                context_window=self.config.llm.context_window,
                thinking=thinking,
            )
        return self._llm_think if thinking else self._llm_fast

    def query(self, question: str, collection: str = "documents", top_k: Optional[int] = None, thinking: bool = True, model: Optional[str] = None) -> QueryResult:
        """Synchronous RAG query."""
        start = time.time()
        try:
            engine = self._get_engine(collection, top_k, thinking, model)
            response = engine.query(question)
            sources = [
                SourceNode(
                    text=n.node.get_content()[:500],
                    score=round(n.score or 0.0, 4),
                    metadata=n.node.metadata,
                )
                for n in (response.source_nodes or [])
            ]
            return QueryResult(
                question=question,
                answer=str(response),
                sources=sources,
                collection=collection,
                elapsed_seconds=round(time.time() - start, 2),
            )
        except Exception as exc:
            raise LLMError(f"Query failed: {exc}") from exc

    async def aquery(self, question: str, collection: str = "documents", top_k: Optional[int] = None, thinking: bool = True, model: Optional[str] = None) -> QueryResult:
        """Async RAG query — use this from FastAPI."""
        start = time.time()
        try:
            engine = self._get_engine(collection, top_k, thinking, model)
            response = await engine.aquery(question)
            sources = [
                SourceNode(
                    text=n.node.get_content()[:500],
                    score=round(n.score or 0.0, 4),
                    metadata=n.node.metadata,
                )
                for n in (response.source_nodes or [])
            ]
            return QueryResult(
                question=question,
                answer=str(response),
                sources=sources,
                collection=collection,
                elapsed_seconds=round(time.time() - start, 2),
            )
        except Exception as exc:
            raise LLMError(f"Async query failed: {exc}") from exc

    def _get_engine(self, collection: str, top_k: Optional[int] = None, thinking: bool = True, model: Optional[str] = None) -> RetrieverQueryEngine:
        model_key = model or self.config.llm.model
        cache_key = f"{collection}:{top_k}:{'think' if thinking else 'fast'}:{model_key}"
        if cache_key in self._engines:
            return self._engines[cache_key]

        try:
            index = self.index_manager.get_index(collection)
        except Exception as exc:
            raise RetrievalError(f"Could not load index for '{collection}': {exc}") from exc

        cfg = self.config.retrieval
        if top_k:
            from copy import deepcopy
            cfg = deepcopy(cfg)
            cfg.similarity_top_k = top_k
            cfg.bm25_top_k = top_k

        retriever = build_retriever(index, cfg)
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=cfg.similarity_cutoff)
        ]
        llm = self._get_llm(thinking, model)
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            llm=llm,
        )
        self._engines[cache_key] = engine
        return engine

    def invalidate_cache(self, collection: str) -> None:
        """Call after ingestion to force engine rebuild."""
        keys = [k for k in self._engines if k.startswith(f"{collection}:")]
        for k in keys:
            del self._engines[k]
        self.index_manager.invalidate_cache(collection)
        logger.info(f"Engine cache cleared for '{collection}'")
