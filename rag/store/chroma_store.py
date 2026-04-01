"""ChromaDB client and collection manager — all chromadb imports live here."""

from __future__ import annotations

import logging
from typing import Dict, List

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.config import AppConfig, ChromaConfig
from rag.exceptions import StoreError

logger = logging.getLogger("rag.store.chroma")


class ChromaManager:
    def __init__(self, config: AppConfig):
        self.config: ChromaConfig = config.chroma
        persist_dir = str(config.abs(self.config.persist_dir))
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB initialised at {persist_dir}")
        except Exception as exc:
            raise StoreError(f"Failed to init ChromaDB: {exc}") from exc
        self._collections: Dict[str, chromadb.Collection] = {}

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        if name in self._collections:
            return self._collections[name]
        try:
            col = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
            self._collections[name] = col
            logger.info(f"Collection '{name}' ready ({col.count()} docs)")
            return col
        except Exception as exc:
            raise StoreError(f"Failed to get/create collection '{name}': {exc}") from exc

    def get_vector_store(self, collection_name: str) -> ChromaVectorStore:
        col = self.get_or_create_collection(collection_name)
        return ChromaVectorStore(chroma_collection=col)

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            logger.info(f"Deleted collection '{name}'")
        except Exception as exc:
            raise StoreError(f"Failed to delete collection '{name}': {exc}") from exc

    def collection_stats(self, name: str) -> dict:
        col = self.get_or_create_collection(name)
        return {
            "name": name,
            "count": col.count(),
            "metadata": col.metadata or {},
        }

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
