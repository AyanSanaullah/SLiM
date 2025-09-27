"""Simple runner without database dependencies for quick testing."""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import json

# Set environment variables for testing
os.environ["DB_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["GEMINI_API_KEY"] = "AIzaSyDXsFUi6NneoeTStVIWKHn6Z1g4oNUTHeo"

app = FastAPI(title="AI Router - Simple Test")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    user_id: str
    prompt: str
    topic_hint: Optional[str] = None
    force_llm: bool = False

class RouteResponse(BaseModel):
    answer: str
    model_type: str
    model_id: str
    topic: str
    routed_reason: str
    training_job_enqueued: bool = False

# Simple topic detection using keywords
def detect_topic_simple(prompt: str, hint: Optional[str] = None) -> str:
    if hint:
        return hint.lower()
    
    prompt_lower = prompt.lower()
    
    topic_keywords = {
        "coding": ["code", "programming", "function", "bug", "debug", "algorithm", "python", "javascript", "sql"],
        "math": ["calculate", "equation", "solve", "formula", "mathematics", "algebra", "geometry"],
        "weather": ["weather", "temperature", "rain", "sunny", "cloudy", "forecast", "climate"],
        "health": ["health", "medical", "doctor", "symptoms", "medicine", "treatment"],
        "travel": ["travel", "vacation", "trip", "hotel", "flight", "destination"],
        "business": ["strategy", "marketing", "sales", "revenue", "profit", "management"]
    }
    
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in prompt_lower)
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    
    return "general"

# Mock LLM response
async def mock_llm_response(prompt: str, topic: str) -> dict:
    """Mock LLM response for testing."""
    
    topic_responses = {
        "coding": f"Here's how to solve your coding question: {prompt[:50]}... [Mock response for {topic}]",
        "math": f"The mathematical solution is: {prompt[:50]}... [Mock calculation for {topic}]",
        "weather": f"The weather information: {prompt[:50]}... [Mock weather data for {topic}]",
        "health": f"Health advice: {prompt[:50]}... [Mock medical info for {topic}]",
        "travel": f"Travel recommendation: {prompt[:50]}... [Mock travel info for {topic}]",
        "business": f"Business insight: {prompt[:50]}... [Mock business advice for {topic}]",
        "general": f"General response: {prompt[:50]}... [Mock general answer]"
    }
    
    return {
        "content": topic_responses.get(topic, topic_responses["general"]),
        "model_id": "mock-llm-v1",
        "latency_ms": 150,
        "cost_usd": 0.001
    }

@app.post("/v1/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest):
    """Route a prompt to appropriate model (mock version)."""
    
    start_time = time.time()
    
    # Detect topic
    topic = detect_topic_simple(request.prompt, request.topic_hint)
    
    # For demo, always route to mock LLM
    response = await mock_llm_response(request.prompt, topic)
    
    return RouteResponse(
        answer=response["content"],
        model_type="llm",
        model_id=response["model_id"],
        topic=topic,
        routed_reason="mock_demo",
        training_job_enqueued=False
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "AI Router API - Simple Test Mode",
        "version": "1.0.0-test",
        "endpoints": {
            "route": "POST /v1/route",
            "health": "GET /health"
        },
        "test_request": {
            "user_id": "test",
            "prompt": "How do I fix this Python bug?",
            "topic_hint": "coding"
        }
    }

if __name__ == "__main__":
    print("Starting AI Router in Simple Test Mode")
    print("API available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    print("Test endpoint: POST /v1/route")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)