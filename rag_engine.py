"""
Core RAG engine - handles document ingestion, retrieval, and querying.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore
import chromadb

from config import RAGConfig

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine with document ingestion, retrieval, and generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        logger.info("Initializing RAG Engine...")

        # Setup LLM
        self.llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            request_timeout=120.0  # 2 minutes timeout
        )

        # Setup Embedding Model
        self.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP

        # Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Setup Document Store
        self.docstore = SimpleDocumentStore()

        # Setup Storage Context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore
        )

        # Setup Node Parser
        self.node_parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        # Setup Ingestion Pipeline with caching
        self.ingestion_cache = IngestionCache()
        self.pipeline = IngestionPipeline(
            transformations=[
                self.node_parser,
                self.embed_model
            ],
            cache=self.ingestion_cache
        )

        # Initialize index (will be populated during ingestion)
        self.index = None
        self._load_or_create_index()

        logger.info("RAG Engine initialized successfully")

    def _load_or_create_index(self):
        """Load existing index or create new one."""
        try:
            # Try to load existing index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            logger.info("Loaded existing vector index")
        except Exception as e:
            logger.info(f"Creating new index: {e}")
            # Create new index
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )

    def ingest_documents(self):
        """Ingest all documents from data directory."""
        try:
            data_dir = self.config.DATA_DIR
            if not data_dir.exists() or not list(data_dir.glob("*")):
                logger.warning(f"No documents found in {data_dir}")
                return

            logger.info(f"Starting document ingestion from {data_dir}...")

            # Load documents
            documents = SimpleDirectoryReader(str(data_dir)).load_data()

            if not documents:
                logger.warning("No documents loaded")
                return

            logger.info(f"Loaded {len(documents)} documents")

            # Run through ingestion pipeline
            nodes = self.pipeline.run(documents=documents)

            logger.info(f"Transformed into {len(nodes)} nodes")

            # Add to index
            if self.index is None:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model
                )
            else:
                self.index.insert_nodes(nodes)

            logger.info(f"Successfully ingested {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            raise

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG engine.

        Args:
            query: Natural language query
            top_k: Number of top results to retrieve
            model: Optional model override

        Returns:
            Dictionary with answer and sources
        """
        try:
            if self.index is None:
                return {
                    "answer": "No documents have been ingested yet. Please upload documents first.",
                    "sources": [],
                    "error": "No documents in vector store"
                }

            top_k = top_k or self.config.TOP_K

            # Use provided model or default
            if model and model != self.config.LLM_MODEL:
                temp_llm = Ollama(
                    model=model,
                    base_url=self.config.OLLAMA_BASE_URL,
                    temperature=self.config.TEMPERATURE,
                    request_timeout=120.0
                )
            else:
                temp_llm = self.llm

            # Create query engine with fusion retrieval (dense + sparse)
            try:
                # Try fusion retrieval (BM25 + dense)
                vector_retriever = self.index.as_retriever(similarity_top_k=top_k)
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=self.docstore,
                    similarity_top_k=top_k
                )

                fusion_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=top_k,
                    mode="relative_score"
                )

                query_engine = self.index.as_query_engine(
                    retriever=fusion_retriever,
                    llm=temp_llm
                )
            except Exception as e:
                logger.warning(f"Fusion retrieval failed, using vector-only: {e}")
                # Fallback to vector-only retrieval
                query_engine = self.index.as_query_engine(
                    similarity_top_k=top_k,
                    llm=temp_llm
                )

            # Query
            response = query_engine.query(query)

            # Extract sources
            sources = []
            if hasattr(response, "source_nodes"):
                for source_node in response.source_nodes:
                    if hasattr(source_node, "metadata") and "file_name" in source_node.metadata:
                        sources.append(source_node.metadata["file_name"])

            return {
                "answer": str(response),
                "sources": list(set(sources)),  # Remove duplicates
                "response_time": getattr(response, "response_time", None)
            }

        except Exception as e:
            logger.error(f"Error during query: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics."""
        return {
            "llm_model": self.config.LLM_MODEL,
            "embed_model": self.config.EMBED_MODEL,
            "documents_count": len(list(self.config.DATA_DIR.glob("*"))),
            "vector_store_size": self.chroma_collection.count() if self.chroma_collection else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
