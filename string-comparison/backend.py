from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import uvicorn
import os
import numpy as np
from typing import List, Dict
# import spacy  # Temporarily disabled due to compatibility issues
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Semantic Comparison API")

# Load the models once at startup
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# nlp = spacy.load("en_core_web_sm")  # Temporarily disabled

def print_interaction_json(interaction_type: str, data: dict):
    """Print detailed JSON for each interaction"""
    interaction_log = {
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,
        "data": data
    }
    
    print("\n" + "="*80)
    print("🔍 STRING COMPARISON INTERACTION")
    print("="*80)
    print(json.dumps(interaction_log, indent=2, ensure_ascii=False))
    print("="*80 + "\n")

# No additional data downloads needed for sentence-level comparison

class ComparisonRequest(BaseModel):
    sentence1: str
    sentence2: str

class ComparisonResponse(BaseModel):
    similarity: float
    sentence1: str
    sentence2: str

def semantic_similarity(a: str, b: str) -> float:
    """Calculate semantic similarity between two sentences"""
    # Encode -> L2-normalized embeddings
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    # Cosine similarity in [-1,1]; with normalized vecs it's the dot product
    sim = torch.matmul(emb[0], emb[1]).item()
    # Map from [-1,1] to [0,1] if you prefer
    return (sim + 1) / 2

# Removed word-level analysis functions - focusing only on sentence-level comparison

@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    return FileResponse('index.html')

@app.post("/compare", response_model=ComparisonResponse)
async def compare_sentences(request: ComparisonRequest):
    """Compare two sentences semantically"""
    try:
        if not request.sentence1.strip() or not request.sentence2.strip():
            raise HTTPException(status_code=400, detail="Both sentences must be provided")
        
        # Log the incoming request
        print_interaction_json("REQUEST_RECEIVED", {
            "sentence1": request.sentence1,
            "sentence2": request.sentence2
        })
        
        similarity = semantic_similarity(request.sentence1, request.sentence2)
        
        # Create response
        response = ComparisonResponse(
            similarity=similarity,
            sentence1=request.sentence1,
            sentence2=request.sentence2
        )
        
        # Log the simplified response
        print_interaction_json("COMPARISON_COMPLETED", {
            "similarity_score": similarity,
            "similarity_percentage": f"{similarity * 100:.2f}%",
            "quality_label": "HIGH" if similarity >= 0.8 else "MEDIUM" if similarity >= 0.5 else "LOW"
        })
        
        return response
        
    except Exception as e:
        # Log the error
        print_interaction_json("ERROR", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "sentence1": request.sentence1 if 'request' in locals() else "N/A",
            "sentence2": request.sentence2 if 'request' in locals() else "N/A"
        })
        raise HTTPException(status_code=500, detail=f"Error processing comparison: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Log health check requests
    print_interaction_json("HEALTH_CHECK", {
        "status": "healthy",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "message": "String Comparison Service is running"
    })
    return {"status": "healthy", "model": "sentence-transformers/all-MiniLM-L6-v2"}

if __name__ == "__main__":
    # Log startup information
    print_interaction_json("SERVICE_STARTUP", {
        "service_name": "String Comparison API",
        "host": "0.0.0.0",
        "port": 8000,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "endpoints": [
            {"method": "GET", "path": "/", "description": "Serve HTML interface"},
            {"method": "POST", "path": "/compare", "description": "Compare two sentences semantically"},
            {"method": "GET", "path": "/health", "description": "Health check endpoint"}
        ],
        "features": [
            "Semantic similarity using sentence transformers",
            "Sentence-level comparison only",
            "Cosine similarity on sentence embeddings",
            "Quality assessment (HIGH/MEDIUM/LOW)",
            "Detailed JSON logging",
            "Fast and focused comparison"
        ]
    })
    
    print("\n🚀 Starting String Comparison Service...")
    print("📊 All interactions will be logged in JSON format")
    print("🔍 Watch for detailed comparison analytics\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
