"""Gemini LLM adapter."""

import os
import time
import httpx
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class GeminiLLM:
    """Gemini API adapter for LLM generation."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Gemini API."""
        start_time = time.time()
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 1024),
            }
        }
        
        params = {"key": self.api_key}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Estimate cost (rough approximation)
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(content.split()) * 1.3
                cost_usd = (input_tokens + output_tokens) * 0.000001  # $0.001 per 1k tokens
                
                logger.info(
                    "gemini_generation_completed",
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd
                )
                
                return {
                    "content": content,
                    "model_id": f"gemini_{self.model}",
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "metadata": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "model": self.model
                    }
                }
                
            except httpx.HTTPError as e:
                logger.error("gemini_api_error", error=str(e))
                raise RuntimeError(f"Gemini API error: {e}")
            except (KeyError, IndexError) as e:
                logger.error("gemini_response_parse_error", error=str(e))
                raise RuntimeError(f"Unexpected Gemini API response format: {e}")
    
    def get_model_id(self) -> str:
        """Get the model identifier."""
        return f"gemini_{self.model}"