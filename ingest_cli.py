"""CLI tool — ingest documents into the RAG pipeline."""

import argparse
import sys
from pathlib import Path

from rag.config import settings
from rag.logging_setup import setup_logging
from rag.store.chroma_store import ChromaManager
from rag.ingest.pipeline import RAGIngestionPipeline

setup_logging(settings)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--collection",
        default=settings.chroma.default_collection,
        help="ChromaDB collection name (default: documents)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if file is unchanged",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: path not found — {path}")
        sys.exit(1)

    chroma = ChromaManager(settings)
    pipeline = RAGIngestionPipeline(settings, chroma)

    if path.is_file():
        result = pipeline.ingest_file(path, collection=args.collection, force=args.force)
    else:
        result = pipeline.ingest_directory(path, collection=args.collection, force=args.force)

    print("\n--- Ingest Result ---")
    print(f"Status        : {result.status}")
    print(f"Files ingested: {result.files_processed}")
    print(f"Files skipped : {result.files_skipped}")
    print(f"Nodes created : {result.nodes_created}")
    print(f"Errors        : {len(result.errors)}")
    print(f"Time          : {result.elapsed_seconds:.1f}s")
    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  - {e}")

    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
