"""LlamaIndex VectorStoreIndex lifecycle manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from llama_index.core import VectorStoreIndex, StorageContext

from rag.config import AppConfig
from rag.store.chroma_store import ChromaManager
from rag.exceptions import StoreError

logger = logging.getLogger("rag.store.index_manager")


class IndexManager:
    def __init__(self, config: AppConfig, chroma_manager: ChromaManager):
        self.config = config
        self.chroma = chroma_manager
        self._indexes: Dict[str, VectorStoreIndex] = {}

    def get_index(self, collection: str = "documents") -> VectorStoreIndex:
        """Return a cached VectorStoreIndex for the given collection."""
        if collection in self._indexes:
            return self._indexes[collection]
        return self._load_index(collection)

    def get_or_create_index(self, collection: str = "documents") -> VectorStoreIndex:
        """Return index, creating empty one if collection has no docs yet."""
        if collection in self._indexes:
            return self._indexes[collection]
        try:
            return self._load_index(collection)
        except StoreError:
            return self._create_empty_index(collection)

    def invalidate_cache(self, collection: str) -> None:
        """Force reload on next access (call after ingestion)."""
        self._indexes.pop(collection, None)
        logger.info(f"Index cache invalidated for '{collection}'")

    def _load_index(self, collection: str) -> VectorStoreIndex:
        vector_store = self.chroma.get_vector_store(collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
            )
            self._indexes[collection] = index
            logger.info(f"Index loaded for collection '{collection}'")
            return index
        except Exception as exc:
            raise StoreError(f"Failed to load index for '{collection}': {exc}") from exc

    def _create_empty_index(self, collection: str) -> VectorStoreIndex:
        vector_store = self.chroma.get_vector_store(collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)
        self._indexes[collection] = index
        logger.info(f"Empty index created for collection '{collection}'")
        return index
