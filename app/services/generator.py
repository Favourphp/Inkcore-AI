# app/services/generator.py
"""
Generator service composes prompt from:
- user request
- retrieved memory (relevant docs)
- style profile
and then uses Groq client to generate text. It also supports Markdown export and JSON.
"""

from app.services.groq_client import get_groq_client
from app.services import memory, analyzer
from typing import List, Dict, Any, Optional
import asyncio
import logging
import markdown2

logger = logging.getLogger(__name__)

async def _retrieve_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top_k relevant documents from memory for context.
    """
    return await memory.query(query, top_k=top_k)

def _compose_prompt(user_prompt: str, contexts: List[Dict[str, Any]], style_profile: Dict[str, Any], constraints: Optional[Dict[str,str]] = None) -> str:
    """
    Compose a single prompt that provides context and instructions to the LLM.
    """
    sb = []
    sb.append("You are a writing assistant helping generate content in the user's style.")
    if style_profile:
        avg_len = style_profile.get("avg_length_words")
        if avg_len:
            sb.append(f"User typical article length (words): {int(avg_len)}")
        common_words = style_profile.get("most_common_words", [])[:20]
        if common_words:
            top_words = ", ".join(w for w,_ in common_words[:10])
            sb.append(f"Frequent words/phrases: {top_words}")
    sb.append("\nPast user content to mimic (most relevant first):")
    for i, ctx in enumerate(contexts[:5]):
        sb.append(f"--- Context {i+1} (id={ctx.get('id')}, metadata={ctx.get('metadata')}):\n{ctx.get('content')[:800]}\n")
    sb.append("\nInstructions:")
    sb.append("Write in the same tone, voice, and structure as the user's past content above.")
    sb.append("Follow the user's prompt exactly. If a target word count is given, aim for that length.")
    if constraints:
        for k,v in constraints.items():
            sb.append(f"Constraint: {k} => {v}")
    sb.append("\nUser prompt:")
    sb.append(user_prompt)
    # Final instruction to the model to output raw markdown or plain text
    sb.append("\nOutput only the final article in plain text. Do NOT include analysis or commentary.")
    return "\n\n".join(sb)

async def generate_blog(user_prompt: str, word_count: int = 1000, model: Optional[str] = None) -> Dict[str,Any]:
    # Retrieve context
    contexts = await _retrieve_context(user_prompt, top_k=8)
    # Build style profile
    style_profile = await analyzer.analyze_documents(contexts)
    # Compose prompt with constraints
    constraints = {"target_word_count": str(word_count)}
    prompt = _compose_prompt(user_prompt + f"\nTarget words: {word_count}", contexts, style_profile, constraints)
    # Call Groq
    client = get_groq_client()
    # set max_tokens roughly: words * 1.4 (tokens~words)
    max_tokens = int(word_count * 1.6) + 100
    try:
        text = await client.generate_text(prompt=prompt, model=model, max_tokens=max_tokens, temperature=0.7)
    except Exception as e:
        logger.exception("Error during generation: %s", e)
        raise

    # Post-process: convert to Markdown (simple)
    md = markdown2.markdown(text)
    return {
        "text": text,
        "markdown": md,
        "metadata": {"model": model or client.model, "word_count_target": word_count}
    }

async def generate_social(user_prompt: str, count: int = 5, platform: str = "linkedin", model: Optional[str] = None) -> Dict[str,Any]:
    contexts = await _retrieve_context(user_prompt, top_k=6)
    style_profile = await analyzer.analyze_documents(contexts)
    results = []
    client = get_groq_client()
    for i in range(count):
        prompt = _compose_prompt(f"{user_prompt}\nCreate a single {platform} post. Keep concise and engaging.", contexts, style_profile, constraints={"post_index": str(i+1)})
        try:
            text = await client.generate_text(prompt=prompt, model=model, max_tokens=200, temperature=0.8)
        except Exception as e:
            logger.exception("Error generating social post #%d: %s", i+1, e)
            text = f"[ERROR generating post {i+1}: {e}]"
        results.append({"text": text})
    # Build markdown as bullet list
    md = "\n\n".join([f"- {r['text']}" for r in results])
    return {"text": "\n\n".join([r["text"] for r in results]), "markdown": md, "metadata": {"count": count, "platform": platform, "model": model or client.model}}
