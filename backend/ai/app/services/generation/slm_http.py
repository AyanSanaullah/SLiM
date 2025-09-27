"""HTTP sLM adapter for remote small language model service."""

import os
import time
import httpx
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger()

class HTTPSLMError(Exception):
    """Exception for HTTP sLM failures."""
    pass

async def generate_slm(prompt: str, topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate response using HTTP sLM service.
    
    Args:
        prompt: User prompt
        topic: Optional topic hint
        
    Returns:
        Dict with answer, model_id, latency_ms, cost_usd
        
    Raises:
        HTTPSLMError: If HTTP call fails or times out
    """
    start_time = time.time()
    
    # Get configuration
    endpoint_url = os.getenv("SLM_ENDPOINT_URL")
    auth_bearer = os.getenv("SLM_AUTH_BEARER")
    timeout_ms = int(os.getenv("SLM_TIMEOUT_MS", "3000"))
    
    if not endpoint_url:
        raise HTTPSLMError("SLM_ENDPOINT_URL not configured")
    
    # Prepare request
    headers = {"Content-Type": "application/json"}
    if auth_bearer:
        headers["Authorization"] = f"Bearer {auth_bearer}"
    
    payload = {
        "prompt": prompt,
        "topic": topic
    }
    
    timeout = httpx.Timeout(timeout_ms / 1000.0)  # Convert to seconds
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("slm_http_request_started", 
                       endpoint=endpoint_url, 
                       topic=topic,
                       timeout_ms=timeout_ms)
            
            response = await client.post(
                endpoint_url,
                json=payload,
                headers=headers
            )
            
            # Check for HTTP errors
            if response.status_code == 401:
                raise HTTPSLMError("Authentication failed")
            elif response.status_code >= 500:
                raise HTTPSLMError(f"Server error: {response.status_code}")
            elif response.status_code >= 400:
                raise HTTPSLMError(f"Client error: {response.status_code}")
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            total_latency_ms = int((time.time() - start_time) * 1000)
            
            # Normalize response format
            result = {
                "content": data.get("answer", ""),
                "model_id": data.get("model_id", "http_slm_unknown"),
                "latency_ms": total_latency_ms,
                "cost_usd": 0.0,  # HTTP sLM has no API cost
                "metadata": {
                    "endpoint": endpoint_url,
                    "topic": topic,
                    "remote_latency_ms": data.get("latency_ms", 0),
                    "total_latency_ms": total_latency_ms
                }
            }
            
            logger.info("slm_http_request_completed",
                       model_id=result["model_id"],
                       latency_ms=total_latency_ms,
                       remote_latency_ms=data.get("latency_ms", 0))
            
            return result
            
    except httpx.TimeoutException:
        logger.error("slm_http_timeout", 
                    endpoint=endpoint_url, 
                    timeout_ms=timeout_ms)
        raise HTTPSLMError("Request timeout")
        
    except httpx.ConnectError:
        logger.error("slm_http_connection_error", endpoint=endpoint_url)
        raise HTTPSLMError("Connection failed")
        
    except httpx.HTTPError as e:
        logger.error("slm_http_error", endpoint=endpoint_url, error=str(e))
        raise HTTPSLMError(f"HTTP error: {e}")
        
    except Exception as e:
        logger.error("slm_http_unexpected_error", endpoint=endpoint_url, error=str(e))
        raise HTTPSLMError(f"Unexpected error: {e}")

async def check_slm_health() -> bool:
    """
    Check if HTTP sLM service is healthy.
    
    Returns:
        True if healthy, False otherwise
    """
    endpoint_url = os.getenv("SLM_ENDPOINT_URL")
    if not endpoint_url:
        return False
    
    # Convert /generate to /healthz
    if endpoint_url.endswith("/generate"):
        health_url = endpoint_url.replace("/generate", "/healthz")
    else:
        health_url = f"{endpoint_url.rstrip('/')}/healthz"
    
    auth_bearer = os.getenv("SLM_AUTH_BEARER")
    timeout_ms = int(os.getenv("SLM_TIMEOUT_MS", "3000"))
    
    headers = {}
    if auth_bearer:
        headers["Authorization"] = f"Bearer {auth_bearer}"
    
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("ok", False)
            
            return False
            
    except Exception as e:
        logger.warning("slm_health_check_failed", error=str(e))
        return False

def is_http_slm_enabled() -> bool:
    """Check if HTTP sLM mode is enabled."""
    slm_mode = os.getenv("SLM_MODE", "local").lower()
    return slm_mode == "http"

def get_slm_config() -> Dict[str, Any]:
    """Get current sLM HTTP configuration."""
    return {
        "mode": os.getenv("SLM_MODE", "local"),
        "endpoint_url": os.getenv("SLM_ENDPOINT_URL", ""),
        "auth_configured": bool(os.getenv("SLM_AUTH_BEARER")),
        "timeout_ms": int(os.getenv("SLM_TIMEOUT_MS", "3000")),
        "enabled": is_http_slm_enabled()
    }