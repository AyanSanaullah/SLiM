"""Main routing endpoint."""

import time
import uuid
import hashlib
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..db.models import Interaction
from ..services.router import RouterService
from utils.observability import get_metrics

router = APIRouter()
metrics = get_metrics()

class RouteRequest(BaseModel):
    user_id: str
    prompt: str
    topic_hint: Optional[str] = None
    force_llm: bool = False
    request_id: Optional[str] = None

class RouteResponse(BaseModel):
    answer: str
    model_type: str
    model_id: str
    topic: str
    routed_reason: str
    training_job_enqueued: bool = False

@router.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest, db: Session = Depends(get_db)):
    """Route a prompt to the appropriate model."""
    
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        # Initialize router service
        router_service = RouterService(db)
        
        # Route the request
        result = await router_service.route_request(
            user_id=request.user_id,
            prompt=request.prompt,
            topic_hint=request.topic_hint,
            force_llm=request.force_llm,
            request_id=request_id
        )
        
        # Record metrics
        latency = time.time() - start_time
        metrics.record_route(
            model_type=result["model_type"],
            topic=result["topic"], 
            latency_seconds=latency
        )
        
        return RouteResponse(**result)
        
    except Exception as e:
        metrics.record_error("routing_error")
        raise HTTPException(status_code=500, detail=str(e))