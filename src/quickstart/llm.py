"""LLM client — a completely from-scratch REST implementation for Gemini.

This bypasses the official Google SDKs to avoid gRPC connection hanging
issues in certain WSL/network environments.

Usage:
    from quickstart import LLMClient

    client = LLMClient()
    reply = client.complete("What is the capital of France?")
    print(reply)
"""

from __future__ import annotations

import logging
import requests
from typing import Any

from quickstart.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Synchronous LLM client using Gemini via REST API."""

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model or settings.model
        self.system = system
        self.api_key = api_key or settings.gemini_api_key.get_secret_value()
        
        # Strip 'models/' prefix if provided
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name[7:]
            
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn completion — returns just the text."""
        
        url = f"{self.base_url}?key={self.api_key}"
        
        # Build payload
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        # Handle system instructions
        sys_prompt = system if system is not None else self.system
        if sys_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": sys_prompt}]
            }
            
        # Generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if "response_mime_type" in kwargs:
            generation_config["responseMimeType"] = kwargs["response_mime_type"]
            
        if generation_config:
            payload["generationConfig"] = generation_config
            
        logger.debug(f"Sending REST request to {self.model_name}")
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract text from response
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract text from response: {data}")
                raise ValueError("Unexpected response structure from Gemini API") from e
                
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API Error: {e.response.status_code} - {e.response.text}")
            logger.error(f"Failed to connect to Gemini API: {e}")
            raise
