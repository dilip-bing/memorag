"""
Full pipeline test — ingests sample docs then queries them.
Run: python pipeline_test.py
"""

from pathlib import Path
from rag.config import settings
from rag.logging_setup import setup_logging
from rag.store.chroma_store import ChromaManager
from rag.store.index_manager import IndexManager
from rag.ingest.pipeline import RAGIngestionPipeline
from rag.query.engine import RAGQueryEngine

setup_logging(settings)

# ── Create sample test documents ──────────────────────────────────────────
test_dir = Path("data/test")
test_dir.mkdir(parents=True, exist_ok=True)

(test_dir / "about_dilip.txt").write_text(
    "Dilip is a software developer based in the US. "
    "He owns an Acer Nitro AN515-58 laptop with an RTX 3070 Ti GPU and 16GB RAM. "
    "He is setting up a local RAG system on his Windows 11 machine."
)
(test_dir / "about_rag.txt").write_text(
    "RAG stands for Retrieval Augmented Generation. "
    "It combines a retrieval system with a language model to answer questions "
    "based on a knowledge base of documents. "
    "The pipeline embeds documents into a vector store and retrieves relevant chunks "
    "at query time to provide accurate, grounded answers."
)
(test_dir / "about_stack.txt").write_text(
    "This RAG stack uses Ollama for local LLM inference, "
    "ChromaDB as the vector database, LlamaIndex as the orchestration framework, "
    "and nomic-embed-text for generating embeddings. "
    "The LLM model is qwen2.5:7b which has a 128K context window."
)

# ── Init components ────────────────────────────────────────────────────────
print("Initialising components...")
chroma = ChromaManager(settings)
index_manager = IndexManager(settings, chroma)
pipeline = RAGIngestionPipeline(settings, chroma)
engine = RAGQueryEngine(settings, index_manager)

# ── Ingest ─────────────────────────────────────────────────────────────────
print("\nIngesting test documents...")
result = pipeline.ingest_directory(test_dir, collection="test", force=True)
print(f"  Files: {result.files_processed} | Nodes: {result.nodes_created} | Time: {result.elapsed_seconds:.1f}s")

# Invalidate cache so engine picks up new nodes
engine.invalidate_cache("test")

# ── Query ──────────────────────────────────────────────────────────────────
questions = [
    "What laptop does Dilip have?",
    "What is RAG and how does it work?",
    "Which LLM model is being used in this stack?",
]

print("\n--- Query Results ---\n")
for q in questions:
    print(f"Q: {q}")
    res = engine.query(q, collection="test")
    print(f"A: {res.answer}")
    print(f"   ({res.elapsed_seconds}s | {len(res.sources)} sources)\n")

print("All tests passed!")
