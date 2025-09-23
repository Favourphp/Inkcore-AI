# app/services/analyzer.py
"""
Content analyzer that inspects past documents to learn:
- common lengths
- typical opening styles
- frequent phrases / words
- sentiment / tone heuristics

This is intentionally lightweight. For deeper style transfer you'd use
fine-tuning or retrieval-augmented generation with richer features.
"""

from typing import List, Dict, Any
from collections import Counter
import re
import asyncio
import logging
from app.services import memory

logger = logging.getLogger(__name__)

async def analyze_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a list of documents (each dict: {id, content, metadata}) and return
    a style profile summarizing typical patterns.
    """
    # Run CPU-bound analysis in thread to avoid blocking if necessary
    return await asyncio.to_thread(_analyze, documents)

def _analyze(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not documents:
        return {"summary": "no_docs", "word_freq": {}, "avg_length": 0}

    lengths = []
    word_counter = Counter()
    opening_phrases = Counter()

    for doc in documents:
        text = doc.get("content", "")
        words = re.findall(r"\w+", text.lower())
        lengths.append(len(words))
        word_counter.update(words)
        # Opening phrase: first 20 words
        first_words = " ".join(words[:20])
        opening_phrases.update([first_words])

    most_common_words = word_counter.most_common(50)
    most_common_openings = opening_phrases.most_common(5)

    profile = {
        "avg_length_words": sum(lengths) / len(lengths),
        "median_length_words": sorted(lengths)[len(lengths)//2],
        "most_common_words": most_common_words,
        "common_openings": most_common_openings,
    }
    logger.debug("Generated style profile: avg_len=%s", profile["avg_length_words"])
    return profile

async def build_style_profile_for_user(top_k: int = 50):
    """
    Pull top_k most relevant/most recent articles from memory and build profile.
    """
    # Here we query memory for a generic "user writing" retrieval — in many setups you might store
    # a dedicated 'articles' collection; for simplification we'll fetch top_k recent documents
    res = await memory.query("user writing", top_k=top_k)
    return await analyze_documents(res)
