# AI Author Agent (FastAPI + Groq + ChromaDB)

## Overview
FastAPI-based AI agent that:
- Integrates with Groq for text generation (async via httpx).
- Stores & retrieves user content in ChromaDB (vector DB).
- Learns style & generates blog posts and social posts.
- Exportable as Markdown/JSON.

## Setup
1. Copy `.env.example` to `.env` and fill in:
   - GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
   - CHROMA_PERSIST_DIR

2. Create virtualenv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
