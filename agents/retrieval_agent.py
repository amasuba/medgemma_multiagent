"""
retrieval_agent.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

try:
    import faiss  # type: ignore
except ImportError:  # FAISS is optional (CPU/GPU split handled by requirements.txt)
    faiss = None  # pragma: no cover

from .base_agent import BaseAgent, AgentNotInitializedError


# --------------------------------------------------------------------------- #
# Helper dataclasses / types
# --------------------------------------------------------------------------- #

Report = Dict[str, Any]  # simple alias for clarity


# --------------------------------------------------------------------------- #
# RetrievalAgent
# --------------------------------------------------------------------------- #


class RetrievalAgent(BaseAgent):
    """
    Retrieves top-k semantically-similar chest-X-ray reports.

    Payload contract
    ----------------
    • query_text   : str   — text query (preferred)
    • vision_analysis : str   — fallback if no explicit query_text
    • top_k        : int   — number of neighbours to return (default = cfg.top_k)
    """

    def __init__(self, cfg: Dict[str, Any], retrieval_cfg: Dict[str, Any]):
        super().__init__(cfg, global_cfg=cfg.get("global", {}))
        self.retrieval_cfg = retrieval_cfg

        # Key parameters
        self.top_k: int = retrieval_cfg.get("top_k", 5)
        self.similarity_threshold: float = retrieval_cfg.get(
            "similarity_threshold", 0.7
        )

        # Embedding
        self.embed_model_name: str = retrieval_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embed_device: str = retrieval_cfg.get("device", "auto")

        # Vector DB settings
        self.vdb_type: str = retrieval_cfg.get("search_methods", ["chromadb"])[0]
        self.persist_dir: Path = Path(
            self.retrieval_cfg.get("vector_database", {})
            .get("persist_directory", "./data/vector_db/persist")
        )
        self.collection_name: str = retrieval_cfg.get(
            "vector_database", {}
        ).get("collection_name", "chest_xray_reports")

        # Internal state
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._faiss_index = None
        self._report_store: List[Report] = []  # raw reports in memory

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    async def _initialize(self) -> None:
        """Load embedding model and open / create vector database."""
        t0 = time.perf_counter()

        # 1) Sentence-Transformers encoder
        self.log.info(f"Loading embedding model: {self.embed_model_name}")
        self._embedder = SentenceTransformer(
            self.embed_model_name, device=self.embed_device
        )

        # 2) Vector DB back-end
        if self.vdb_type.lower() == "chromadb":
            self.log.info("Initialising ChromaDB backend")
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            if self.collection_name not in [
                c.name for c in self._chroma_client.list_collections()
            ]:
                self._collection = self._chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.embed_model_name
                    ),
                )
            else:
                self._collection = self._chroma_client.get_collection(
                    self.collection_name
                )
        elif self.vdb_type.lower() == "faiss":
            if faiss is None:
                raise RuntimeError("FAISS selected but faiss-cpu/faiss-gpu not installed")
            self.log.info("Initialising FAISS backend (in-memory)")
            self._faiss_index = None  # Lazy build with first ingest
        else:
            raise ValueError(f"Unsupported vector DB type: {self.vdb_type}")

        self.log.info(f"RetrievalAgent ready in {(time.perf_counter()-t0):.2f}s")

    async def _shutdown(self) -> None:
        """Flush and close vector database connections."""
        if self._chroma_client:
            self.log.debug("Closing ChromaDB client")
            self._chroma_client.persist()
            self._chroma_client = None
        self.log.debug("RetrievalAgent shutdown complete")

    # --------------------------------------------------------------------- #
    # Core retrieval logic
    # --------------------------------------------------------------------- #

    async def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._embedder is None:
            raise AgentNotInitializedError("RetrievalAgent not initialised")

        query_text: str = payload.get("query_text") or payload.get("vision_analysis")
        if not query_text:
            raise ValueError("Payload must include 'query_text' or 'vision_analysis'")

        top_k = int(payload.get("top_k", self.top_k))

        # Encode query
        query_vec = self._embedder.encode(query_text, normalize_embeddings=True)

        # Search
        if self.vdb_type.lower() == "chromadb":
            assert self._collection is not None
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
            )
            ids = results["ids"][0]
            distances = results["distances"][0]
            matched_reports = [
                {
                    **self._report_store[int(idx)],
                    "similarity": 1.0 - dist,
                }
                for idx, dist in zip(ids, distances)
                if (1.0 - dist) >= self.similarity_threshold
            ]
        else:  # FAISS
            if self._faiss_index is None or not self._report_store:
                self.log.warning("FAISS index empty — returning no matches")
                matched_reports = []
            else:
                D, I = self._faiss_index.search(query_vec.reshape(1, -1), top_k)
                matched_reports = [
                    {
                        **self._report_store[int(idx)],
                        "similarity": 1.0 - float(dist),
                    }
                    for idx, dist in zip(I[0], D[0])
                    if (1.0 - dist) >= self.similarity_threshold
                ]

        return {
            "matched_reports": matched_reports,
            "num_matches": len(matched_reports),
            "top_k": top_k,
        }

    # --------------------------------------------------------------------- #
    # Public helper — knowledge-base ingestion
    # --------------------------------------------------------------------- #

    async def add_reports(self, reports: List[Report]) -> None:
        """
        Add new reports to the vector store.

        Each report dict **must** contain at least:
            • 'report_id' : str
            • 'content'   : str  (plain text)
        Additional metadata will be preserved.
        """
        if self._embedder is None:
            raise AgentNotInitializedError("RetrievalAgent not initialised")

        if not reports:
            self.log.warning("add_reports() called with empty list")
            return

        texts = [r["content"] for r in reports]
        self.log.info(f"Embedding {len(texts)} new reports")
        embeddings = self._embedder.encode(
            texts, batch_size=64, normalize_embeddings=True
        )

        # Store raw reports for quick lookup
        start_idx = len(self._report_store)
        self._report_store.extend(reports)

        if self.vdb_type.lower() == "chromadb":
            assert self._collection is not None
            self._collection.add(
                ids=[str(i) for i in range(start_idx, start_idx + len(texts))],
                documents=texts,
                embeddings=list(embeddings),
                metadatas=[{k: v for k, v in r.items() if k not in {"content"}} for r in reports],
            )
            self._collection.persist()
        else:  # FAISS
            import numpy as np

            vec_dim = embeddings.shape[1]
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(vec_dim)
            self._faiss_index.add(np.asarray(embeddings, dtype="float32"))

        self.log.success(f"{len(texts)} reports ingested into vector store ✔")

    # --------------------------------------------------------------------- #
    # Utility / diagnostics
    # --------------------------------------------------------------------- #

    def _info(self) -> Dict[str, Any]:
        """Return internal diagnostics (for orchestrator dashboards)."""
        backend = self.vdb_type.upper()
        size = len(self._report_store)
        return {
            "backend": backend,
            "db_size": size,
            "embed_model": self.embed_model_name,
            "top_k_default": self.top_k,
        }


# --------------------------------------------------------------------------- #
# Convenience CLI (python -m medgemma_multiagent.agents.retrieval_agent ...)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Standalone retrieval test")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", required=True, help="Free-text query")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Minimal ad-hoc run without orchestrator
    agent_cfg: Dict[str, Any] = {
        "name": "RetrievalAgent",
        "description": "Standalone test instance",
    }
    retrieval_cfg: Dict[str, Any] = {
        "top_k": args.top_k,
    }
    agent = RetrievalAgent(agent_cfg, retrieval_cfg)

    async def _demo() -> None:
        await agent.initialize()
        result = await agent.process({"query_text": args.query, "top_k": args.top_k})
        print(json.dumps(result, indent=2))
        await agent.shutdown()

    asyncio.run(_demo())
