"""RAG query engine — retrieves context and generates answers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncGenerator

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.ollama import Ollama

# ── Relevance threshold ────────────────────────────────────────────────────
# If the best-matching document scores below this, the question is unrelated
# to the knowledge base → skip RAG and answer from general LLM knowledge.
GENERAL_KNOWLEDGE_THRESHOLD = 0.40

# ── Smart QA prompt ────────────────────────────────────────────────────────
# Allows the LLM to use its own knowledge when documents aren't relevant.
_QA_PROMPT = PromptTemplate(
    "You are a knowledgeable AI assistant.\n\n"
    "Document context (use if relevant to the question):\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Instructions:\n"
    "  - If the document context directly addresses the question, use it.\n"
    "  - If the context is NOT relevant, answer from your own training knowledge.\n"
    "  - Never say 'the context does not contain' — just answer the question.\n"
    "  - Be direct and concise.\n\n"
    "Answer:"
)

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

    def _check_relevance(self, engine: RetrieverQueryEngine, question: str) -> float:
        """Return the max retrieval score for the question. 0.0 if no nodes."""
        try:
            nodes = engine.retriever.retrieve(question)
            return max((n.score or 0.0 for n in nodes), default=0.0)
        except Exception:
            return 0.0

    async def _acheck_relevance(self, engine: RetrieverQueryEngine, question: str) -> float:
        """Async version of _check_relevance."""
        try:
            nodes = await engine.retriever.aretrieve(question)
            return max((n.score or 0.0 for n in nodes), default=0.0)
        except Exception:
            return 0.0

    def _direct_llm_result(self, question: str, llm: Ollama, collection: str, elapsed: float) -> QueryResult:
        """Answer directly from LLM knowledge, no document context."""
        response = llm.complete(question)
        return QueryResult(
            question=question,
            answer=str(response),
            sources=[],
            collection=collection,
            elapsed_seconds=elapsed,
        )

    async def _adirect_llm_result(self, question: str, llm: Ollama, collection: str, start: float) -> QueryResult:
        """Async direct LLM answer, no document context."""
        response = await llm.acomplete(question)
        return QueryResult(
            question=question,
            answer=str(response),
            sources=[],
            collection=collection,
            elapsed_seconds=round(time.time() - start, 2),
        )

    def query(self, question: str, collection: str = "documents", top_k: Optional[int] = None, thinking: bool = True, model: Optional[str] = None) -> QueryResult:
        """Synchronous RAG query with automatic general-knowledge fallback."""
        start = time.time()
        try:
            engine = self._get_engine(collection, top_k, thinking, model)
            llm = self._get_llm(thinking, model)

            max_score = self._check_relevance(engine, question)
            if max_score < GENERAL_KNOWLEDGE_THRESHOLD:
                logger.info(f"General-knowledge mode (max_score={max_score:.3f} < {GENERAL_KNOWLEDGE_THRESHOLD})")
                return self._direct_llm_result(question, llm, collection, round(time.time() - start, 2))

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
        """Async RAG query with automatic general-knowledge fallback."""
        start = time.time()
        try:
            engine = self._get_engine(collection, top_k, thinking, model)
            llm = self._get_llm(thinking, model)

            max_score = await self._acheck_relevance(engine, question)
            if max_score < GENERAL_KNOWLEDGE_THRESHOLD:
                logger.info(f"General-knowledge mode (max_score={max_score:.3f} < {GENERAL_KNOWLEDGE_THRESHOLD})")
                return await self._adirect_llm_result(question, llm, collection, start)

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

    async def aquery_stream(
        self,
        question: str,
        collection: str = "documents",
        top_k: Optional[int] = None,
        thinking: bool = True,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream RAG query with status updates and tokens."""
        start = time.time()
        try:
            status_label = "Thinking" if thinking else "Fast thinking"
            yield {"type": "status", "status": status_label, "phase": "retrieving"}

            engine = self._get_engine(collection, top_k, thinking, model)
            llm = self._get_llm(thinking, model)

            # Relevance check — bypass RAG for general knowledge questions
            max_score = await self._acheck_relevance(engine, question)
            if max_score < GENERAL_KNOWLEDGE_THRESHOLD:
                logger.info(f"Stream general-knowledge mode (max_score={max_score:.3f})")
                yield {"type": "status", "status": status_label, "phase": "generating"}
                gk = await llm.acomplete(question)
                yield {"type": "answer", "content": str(gk)}
                yield {"type": "sources", "sources": []}
                yield {"type": "done", "elapsed_seconds": round(time.time() - start, 2), "collection": collection}
                return

            yield {"type": "status", "status": status_label, "phase": "generating"}

            # Stream the response
            streaming_response = await engine.aquery(question)
            
            # Extract sources
            sources = [
                SourceNode(
                    text=n.node.get_content()[:500],
                    score=round(n.score or 0.0, 4),
                    metadata=n.node.metadata,
                )
                for n in (streaming_response.source_nodes or [])
            ]

            # Send answer as complete (LlamaIndex doesn't expose token streaming easily)
            answer = str(streaming_response)
            yield {
                "type": "answer",
                "content": answer,
            }

            # Send sources
            yield {
                "type": "sources",
                "sources": [
                    {"text": s.text, "score": s.score, "metadata": s.metadata}
                    for s in sources
                ],
            }

            # Send completion
            elapsed = round(time.time() - start, 2)
            yield {
                "type": "done",
                "elapsed_seconds": elapsed,
                "collection": collection,
            }

        except Exception as exc:
            logger.error(f"Stream query failed: {exc}", exc_info=True)
            yield {"type": "error", "error": str(exc)}

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
            response_mode="compact",
        )
        # Override the default "answer ONLY from context" prompt with one
        # that lets the LLM fall back to its own knowledge when docs are not relevant.
        engine.update_prompts({"response_synthesizer:text_qa_template": _QA_PROMPT})
        self._engines[cache_key] = engine
        return engine

    def invalidate_cache(self, collection: str) -> None:
        """Call after ingestion to force engine rebuild."""
        keys = [k for k in self._engines if k.startswith(f"{collection}:")]
        for k in keys:
            del self._engines[k]
        self.index_manager.invalidate_cache(collection)
        logger.info(f"Engine cache cleared for '{collection}'")
