# app/services/groq_client.py
"""
Async Groq client wrapper using httpx.AsyncClient.

This module exposes:
- async def generate_text(prompt, model, max_tokens, temperature)
It uses settings from app.config but allows model override.
"""

import httpx
from typing import Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class GroqClientError(Exception):
    pass

class GroqClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or settings.groq_api_key
        self.base_url = base_url or settings.groq_base_url
        self.model = model or settings.groq_model
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.debug("Initialized GroqClient with model=%s base_url=%s", self.model, self.base_url)

    async def close(self):
        await self._client.aclose()

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        stop: Optional[list] = None,
        **kwargs
    ) -> str:
        model_to_use = model or self.model
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            payload["stop"] = stop

        try:
            resp = await self._client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            logger.exception("Network error when calling Groq: %s", exc)
            raise GroqClientError(f"RequestError: {exc}") from exc

        if resp.status_code >= 400:
            raise GroqClientError(f"Groq API error {resp.status_code}: {resp.text}")

        data = resp.json()

        # Extract assistant text
        text = None
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0].get("text") or data["choices"][0].get("output")
            elif "output" in data:
                text = " ".join(map(str, data["output"]))
            elif "text" in data:
                text = data["text"]
            elif "generated_text" in data:
                text = data["generated_text"]

        if text is None:
            raise GroqClientError(f"Could not extract assistant content from response: {data}")

        return text.strip()


# Singleton client
_groq_client: Optional[GroqClient] = None

def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client

async def close_groq_client():
    global _groq_client
    if _groq_client is not None:
        await _groq_client.close()
        _groq_client = None
