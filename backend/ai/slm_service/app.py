"""
FastAPI sLM service for Hugging Face Spaces.
Minimal small language model HTTP service with placeholder model.
"""

import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer
from pydantic import BaseModel


# Configuration from environment
MODEL_ID = os.getenv("MODEL_ID", "slm-topic-general-v1")
AUTH_BEARER = os.getenv("SLM_AUTH_BEARER")

# Security setup
security = HTTPBearer(auto_error=False) if AUTH_BEARER else None

app = FastAPI(
    title="Simple sLM Service",
    description="Minimal small language model HTTP service",
    version="1.0.0"
)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    topic: Optional[str] = None

class GenerateResponse(BaseModel):
    answer: str
    model_id: str
    latency_ms: int

class HealthResponse(BaseModel):
    ok: bool
    model_id: str

# TODO: Replace with real model loading
def load_model():
    """Load the sLM model. Currently a placeholder."""
    return {
        "model": "placeholder",
        "topic_specializations": {
            "coding": "I'm specialized for coding questions",
            "math": "I'm specialized for math problems", 
            "writing": "I'm specialized for writing assistance",
            "general": "I'm a general purpose assistant"
        }
    }

# Global model instance
model_instance = load_model()

def check_auth(credentials = Security(security)) -> bool:
    """Check bearer token if authentication is enabled."""
    if not AUTH_BEARER:
        return True  # No auth required
    
    if not credentials or credentials.credentials != AUTH_BEARER:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication token"
        )
    return True

def generate_response(prompt: str, topic: Optional[str] = None) -> str:
    """
    Generate response using the sLM model.
    TODO: Replace with actual model inference.
    """
    specializations = model_instance["topic_specializations"]
    
    if topic and topic in specializations:
        prefix = specializations[topic]
    else:
        prefix = specializations["general"]
    
    # Placeholder response generation
    if topic == "coding":
        return f"{prefix}: For your coding question '{prompt[:50]}...', here's a concise solution with best practices."
    elif topic == "math":
        return f"{prefix}: Let me solve this step-by-step: '{prompt[:50]}...'"
    elif topic == "writing":
        return f"{prefix}: I'll help improve your text: '{prompt[:50]}...'"
    else:
        return f"{prefix}: Regarding '{prompt[:50]}...', here's my response based on efficient reasoning."

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(ok=True, model_id=MODEL_ID)

@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    authenticated: bool = Depends(check_auth)
):
    """Generate response using the sLM."""
    start_time = time.time()
    
    try:
        # Generate response
        answer = generate_response(request.prompt, request.topic)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return GenerateResponse(
            answer=answer,
            model_id=MODEL_ID,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Simple sLM",
        "model_id": MODEL_ID,
        "endpoints": ["/healthz", "/generate"],
        "auth_required": bool(AUTH_BEARER)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)