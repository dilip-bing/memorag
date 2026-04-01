"""Hybrid retriever — vector + BM25 fusion, or either alone."""

from __future__ import annotations

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from rag.config import RetrievalConfig

logger = logging.getLogger("rag.retrieve.hybrid")


def build_retriever(index: VectorStoreIndex, config: RetrievalConfig):
    """Return the appropriate retriever based on config.retrieval.mode."""
    mode = config.mode.lower()

    if mode == "vector":
        logger.info(f"Using vector retriever (top_k={config.similarity_top_k})")
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=config.similarity_top_k,
        )

    if mode == "bm25":
        nodes = list(index.docstore.docs.values()) if index.docstore else []
        if not nodes:
            logger.warning("BM25: no nodes in docstore, falling back to vector")
            return VectorIndexRetriever(
                index=index,
                similarity_top_k=config.similarity_top_k,
            )
        logger.info(f"Using BM25 retriever ({len(nodes)} nodes, top_k={config.bm25_top_k})")
        return BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=config.bm25_top_k,
        )

    # Default: hybrid
    logger.info(
        f"Using hybrid retriever (vector top_k={config.similarity_top_k}, "
        f"bm25 top_k={config.bm25_top_k})"
    )
    vector_ret = VectorIndexRetriever(
        index=index,
        similarity_top_k=config.similarity_top_k,
    )

    nodes = list(index.docstore.docs.values()) if index.docstore else []
    if not nodes:
        logger.warning("Hybrid: no docstore nodes, using vector-only")
        return vector_ret

    bm25_ret = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=config.bm25_top_k,
    )

    return QueryFusionRetriever(
        retrievers=[vector_ret, bm25_ret],
        similarity_top_k=config.similarity_top_k,
        num_queries=config.num_queries,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=False,
    )
