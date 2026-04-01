"""CLI tool — interactive REPL for querying the RAG pipeline."""

import argparse
import sys

from rag.config import settings
from rag.logging_setup import setup_logging
from rag.store.chroma_store import ChromaManager
from rag.store.index_manager import IndexManager
from rag.query.engine import RAGQueryEngine

setup_logging(settings)


def main():
    parser = argparse.ArgumentParser(description="Query your RAG pipeline")
    parser.add_argument(
        "--collection",
        default=settings.chroma.default_collection,
        help="ChromaDB collection to query",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default=None,
        help="Override retrieval mode from config",
    )
    args = parser.parse_args()

    if args.mode:
        settings.retrieval.mode = args.mode

    chroma = ChromaManager(settings)
    index_manager = IndexManager(settings, chroma)
    engine = RAGQueryEngine(settings, index_manager)

    print(f"\nRAG Query REPL | collection: '{args.collection}' | mode: {settings.retrieval.mode}")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            sys.exit(0)

        try:
            result = engine.query(question, collection=args.collection)
            print(f"\nAnswer: {result.answer}")
            if result.sources:
                print(f"\nSources ({len(result.sources)}):")
                for i, src in enumerate(result.sources, 1):
                    print(f"  [{i}] score={src.score} | {src.metadata.get('file_name', 'unknown')}")
                    print(f"      {src.text[:150]}...")
            print(f"\n({result.elapsed_seconds}s)\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
