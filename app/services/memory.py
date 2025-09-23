"""
Memory service using ChromaDB as a vector store.

This service supports:
- add(prompt, response)
- query(query_text, top_k)
- delete(id)
"""

import chromadb
from typing import Optional, List, Dict, Any
import uuid
import asyncio
import logging
from app.config import settings
import numpy as np

logger = logging.getLogger(__name__)

# Initialize chroma client lazily
_client = None
_collection_name = "user_content"


def _ensure_client():
    """Initialize persistent Chroma client (new API)."""
    global _client
    if _client is None:
        persist_dir = settings.chroma_persist_dir
        _client = chromadb.PersistentClient(path=persist_dir)
        logger.info("Initialized Chroma PersistentClient with persist_dir=%s", persist_dir)
    return _client


async def _run_in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


# ✅ NEW: Friendly `add` wrapper to rhyme with generate.py
async def add(prompt: str, response: str):
    """
    Store a prompt/response pair in memory.
    """
    metadata = {"prompt": prompt}
    content = response
    return await save_document(id=None, content=content, metadata=metadata)


async def save_document(
    id: Optional[str],
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    embed: Optional[List[float]] = None,
):
    """
    Save a document into Chroma. If embed is None, use a simple text embedding fallback
    (replace with a proper embedding model in production).
    """
    client = _ensure_client()

    # Create or get collection
    coll = client.get_or_create_collection(name=_collection_name, metadata={"hnsw:space": "cosine"})

    if id is None:
        id = str(uuid.uuid4())

    metadata = metadata or {}

    # If embed provided, use it. Otherwise compute fallback embedding
    vec = embed if embed is not None else _simple_text_to_vector(content)

    def _add():
        coll.add(documents=[content], metadatas=[metadata], ids=[id], embeddings=[list(vec)])

    await _run_in_thread(_add)
    logger.debug("Saved document id=%s len=%d", id, len(content))
    return id


async def query(query_text: str, top_k: int = 5):
    """
    Query relevant documents using simple vector retrieval.
    """
    client = _ensure_client()
    coll = client.get_or_create_collection(name=_collection_name, metadata={"hnsw:space": "cosine"})

    # Compute embedding for query_text
    qvec = _simple_text_to_vector(query_text)

    def _query():
        return coll.query(
            query_embeddings=[list(qvec)],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

    res = await _run_in_thread(_query)

    results = []
    if res and "documents" in res and len(res["documents"]) > 0:
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for i, content in enumerate(docs):
            results.append(
                {
                    "id": metas[i].get("id", str(i)),
                    "content": content,
                    "metadata": metas[i],
                    "distance": float(dists[i]) if dists is not None else None,
                }
            )
    return results


async def delete(id: str):
    client = _ensure_client()
    coll = client.get_or_create_collection(name=_collection_name, metadata={"hnsw:space": "cosine"})

    def _del():
        coll.delete(ids=[id])

    await _run_in_thread(_del)
    return True


def _simple_text_to_vector(text: str, dim: int = 1536):
    """
    Deterministic but simple text->vector mapping: hash chunks into a vector.
    Replace this with a real embedding model in production.
    """
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)

    # Expand/repeat to requested dim
    if arr.size < dim:
        reps = int(np.ceil(dim / arr.size))
        arr = np.tile(arr, reps)[:dim]
    else:
        arr = arr[:dim]

    # Normalize
    v = arr / (np.linalg.norm(arr) + 1e-8)
    return v.tolist()
