"""Multi-format document loader."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from rag.config import IngestionConfig
from rag.exceptions import DocumentLoadError

logger = logging.getLogger("rag.ingest.loader")


class DocumentLoader:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.supported = set(config.supported_extensions)

    def load_file(self, path: Path) -> List[Document]:
        """Load a single file into Documents with metadata."""
        path = Path(path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if path.suffix.lower() not in self.supported:
            raise DocumentLoadError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {self.supported}"
            )
        try:
            reader = SimpleDirectoryReader(input_files=[str(path)])
            docs = reader.load_data()
            now = datetime.now(timezone.utc).isoformat()
            for doc in docs:
                doc.metadata.update(
                    {
                        "source_path": str(path),
                        "file_name": path.name,
                        "file_type": path.suffix.lower(),
                        "ingested_at": now,
                    }
                )
            logger.info(f"Loaded {len(docs)} doc(s) from {path.name}")
            return docs
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Failed to parse {path}: {exc}") from exc

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> List[Document]:
        """Load all supported files from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise DocumentLoadError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.supported
        ]

        if not files:
            logger.warning(f"No supported files found in {directory}")
            return []

        all_docs: List[Document] = []
        for f in files:
            try:
                all_docs.extend(self.load_file(f))
            except DocumentLoadError as exc:
                logger.error(f"Skipping {f.name}: {exc}")

        logger.info(
            f"Loaded {len(all_docs)} total doc(s) from {len(files)} file(s)"
        )
        return all_docs
