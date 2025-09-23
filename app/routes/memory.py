# app/routes/memory.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import MemorySaveRequest, MemoryQueryRequest, MemoryQueryResponse, MemoryQueryResult
from app.services import memory
import logging
from typing import List

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger(__name__)

@router.post("/save")
async def save_memory(req: MemorySaveRequest):
    """
    Save new content or conversation to memory.
    """
    try:
        doc_id = await memory.save_document(req.id, req.content, metadata={**(req.metadata or {}), "content_type": req.content_type}, embed=req.embed)
        return {"id": doc_id}
    except Exception as e:
        logger.exception("Failed to save memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=MemoryQueryResponse)
async def query_memory(req: MemoryQueryRequest):
    """
    Query memory for relevant past articles.
    """
    try:
        results = await memory.query(req.query, top_k=req.top_k)
        logger.info("Memory query raw results: %s", results)

        mapped = [MemoryQueryResult(id=r["id"], content=r["content"], distance=r.get("distance"), metadata=r.get("metadata")) for r in results]
        return {"results": mapped}
    except Exception as e:
        logger.exception("Failed to query memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
