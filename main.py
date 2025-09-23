# app/main.py
"""
FastAPI entrypoint. Launch with:
uvicorn app.main:app --reload
"""

import logging
import uvicorn
from fastapi import FastAPI
from app.config import settings
from app.routes import generate, memory
from app.services.groq_client import get_groq_client, close_groq_client
import asyncio

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Author Agent", version="0.1.0", description="FastAPI app using Groq + ChromaDB to generate content in user's style.")

# Include routers
app.include_router(generate.router)
app.include_router(memory.router)

@app.on_event("startup")
async def startup():
    logger.info("Starting up app and initializing Groq client...")
    # Ensure groq client initialized
    get_groq_client()
    # Ensure chroma client initialized
    from app.services.memory import _ensure_client as ensure_chroma
    ensure_chroma()
    logger.info("Startup complete.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down, closing Groq client...")
    await close_groq_client()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
