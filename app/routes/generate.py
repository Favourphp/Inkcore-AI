# app/routes/generate.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import GenerateBlogRequest, GenerateResponse, GenerateSocialRequest
from app.services.groq_client import get_groq_client
from app.services import memory, analyzer
from collections import defaultdict
import logging

router = APIRouter(prefix="/generate", tags=["generate"])
logger = logging.getLogger(__name__)

# Separate histories for each type and each user
blog_histories = defaultdict(list)   # user_id -> blog exchanges
social_histories = defaultdict(list) # user_id -> social exchanges

MAX_HISTORY = 20

# ---------------- BLOG ---------------- #
@router.post("/blog")
async def generate_blog_route(request: GenerateBlogRequest):
    """
    Generate long-form blog content.
    User-specific memory + conversation history.
    """
    try:
        client = get_groq_client()

        # --- 1. Retrieve user-specific long-term memory ---
        past_context = memory.query(f"{request.user_id}:{request.prompt}", top_k=3)

        # --- 2. Include this user's blog history ---
        history = blog_histories[request.user_id]
        conversation_snippets = "\n".join(history[-5:])

        # --- 3. Build blog-specific system prompt ---
        full_prompt = (
            "You are a professional blog writer.\n"
            "Your ONLY job is to create engaging, structured, long-form blog posts.\n"
            "Do not answer general questions. If the request is off-topic, "
            "politely explain that you can only help with blog writing.\n\n"
            "Follow this structure:\n"
            "- Catchy introduction (hook)\n"
            "- Main sections with clear headings\n"
            "- Engaging examples and explanations\n"
            "- Concise conclusion with a call-to-action\n\n"
            f"Past posts/patterns:\n{past_context}\n\n"
            f"Recent conversation:\n{conversation_snippets}\n\n"
            f"User request:\n{request.prompt}"
        )

        # --- 4. Call LLM ---
        text = await client.generate_text(
            prompt=full_prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # --- 5. Update user-specific memory + history ---
        memory.add(f"{request.user_id}:{request.prompt}", text)
        history.append(f"User: {request.prompt}\nAI: {text}")

        # Trim history
        if len(history) > MAX_HISTORY:
            history.pop(0)

        return {
            "text": text,
            "markdown": text,
            "metadata": {"source": "groq", "model": request.model},
        }

    except Exception as e:
        logger.exception("Failed to generate blog: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- SOCIAL ---------------- #
@router.post("/social")
async def generate_social_route(request: GenerateSocialRequest):
    """
    Generate short-form social content (tweets, LinkedIn posts).
    User-specific memory + conversation history.
    """
    try:
        client = get_groq_client()

        # --- 1. Retrieve user-specific long-term memory ---
        past_context = memory.query(f"{request.user_id}:{request.prompt}", top_k=3)

        # --- 2. Fetch this user's conversation history ---
        history = social_histories[request.user_id]
        conversation_snippets = "\n".join(history[-5:])

        # --- 3. Build prompt ---
        full_prompt = (
           "You are a **social media strategist and content creator**.\n"
           "Your ONLY job is to generate **short-form social content** "
           "(tweets, LinkedIn posts, Instagram captions, content ideas).\n"
           "You must NEVER answer general knowledge questions or unrelated topics.\n"
           "If the user asks something outside social media content, politely remind them "
           "that you can only create posts and ideas.\n\n"
           "Use the following context to maintain style and continuity:\n\n"
           f"Past posts/patterns:\n{past_context}\n\n"
           f"Recent conversation:\n{conversation_snippets}\n\n"
           f"User request:\n{request.prompt}"
        )

        # --- 4. Call LLM ---
        text = await client.generate_text(
            prompt=full_prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # --- 5. Update memory + history ---
        memory.add(f"{request.user_id}:{request.prompt}", text)
        history.append(f"User: {request.prompt}\nAI: {text}")

        # Trim history
        if len(history) > MAX_HISTORY:
            history.pop(0)

        return {
            "text": text,
            "markdown": text,
            "metadata": {"source": "groq", "model": request.model}
        }

    except Exception as e:
        logger.exception("Failed to generate social content: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
