"""Main router service for prompt routing and repetition detection."""

import os
import time
import hashlib
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from ..db.models import Interaction, ModelRegistry, TrainingJob
from .generation.llm_gemini import GeminiLLM
from .generation.slm_local import LocalSLM
from .generation.slm_http import generate_slm, HTTPSLMError, is_http_slm_enabled
from .similarity import SimilarityService
from .training import TrainingService
import structlog

logger = structlog.get_logger()

class RouterService:
    """Main service for routing prompts and managing the AI pipeline."""
    
    def __init__(self, db: Session):
        self.db = db
        self.similarity_service = SimilarityService()
        self.training_service = TrainingService(db)
        self.gemini_llm = GeminiLLM()
        
        # Configuration from environment
        self.force_llm = os.getenv("ROUTER_FORCE_LLM", "false").lower() == "true"
        self.disable_slm = os.getenv("ROUTER_DISABLE_SLM", "false").lower() == "true"
        self.repetition_threshold = float(os.getenv("REPETITION_SIM_THRESHOLD", "0.90"))
        
    async def route_request(self, 
                          user_id: str, 
                          prompt: str, 
                          topic_hint: Optional[str] = None,
                          force_llm: bool = False,
                          request_id: str = None) -> Dict[str, Any]:
        """Route a request to the appropriate model and handle training logic."""
        
        start_time = time.time()
        
        # Detect topic
        topic = self.similarity_service.detect_topic(prompt, topic_hint)
        logger.info("topic_detected", topic=topic, hint_provided=bool(topic_hint))
        
        # Check for forced routing
        if self.force_llm or force_llm:
            return await self._route_to_llm(user_id, prompt, topic, request_id, "forced_llm")
        
        if self.disable_slm:
            return await self._route_to_llm(user_id, prompt, topic, request_id, "slm_disabled")
        
        # Check if we have a ready sLM for this topic
        ready_slm = self._find_ready_slm(topic)
        if ready_slm:
            return await self._route_to_slm(user_id, prompt, topic, request_id, ready_slm)
        
        # Route to LLM and check for repetition
        result = await self._route_to_llm(user_id, prompt, topic, request_id, "no_slm_available")
        
        # Check for repetition and potentially enqueue training
        training_enqueued = await self._check_and_handle_repetition(
            user_id, prompt, result["answer"], topic
        )
        result["training_job_enqueued"] = training_enqueued
        
        return result
    
    async def _route_to_llm(self, 
                          user_id: str, 
                          prompt: str, 
                          topic: str, 
                          request_id: str,
                          reason: str) -> Dict[str, Any]:
        """Route to LLM and persist interaction."""
        
        logger.info("routing_to_llm", topic=topic, reason=reason)
        
        # Generate response
        response = await self.gemini_llm.generate(prompt)
        
        # Store interaction
        await self._store_interaction(
            user_id=user_id,
            request_id=request_id,
            prompt=prompt,
            output=response["content"],
            topic=topic,
            model_type="llm",
            model_id=response["model_id"],
            latency_ms=response["latency_ms"],
            cost_usd=response["cost_usd"],
            routed_reason=reason
        )
        
        return {
            "answer": response["content"],
            "model_type": "llm",
            "model_id": response["model_id"],
            "topic": topic,
            "routed_reason": reason,
            "training_job_enqueued": False
        }
    
    async def _route_to_slm(self, 
                          user_id: str, 
                          prompt: str, 
                          topic: str, 
                          request_id: str,
                          model_registry: ModelRegistry) -> Dict[str, Any]:
        """Route to sLM and persist interaction."""
        
        logger.info("routing_to_slm", topic=topic, model_id=model_registry.id)
        
        # Try HTTP sLM first if enabled
        if is_http_slm_enabled():
            try:
                response = await generate_slm(prompt, topic)
                
                # Store interaction
                await self._store_interaction(
                    user_id=user_id,
                    request_id=request_id,
                    prompt=prompt,
                    output=response["content"],
                    topic=topic,
                    model_type="slm",
                    model_id=response["model_id"],
                    latency_ms=response["latency_ms"],
                    cost_usd=response["cost_usd"],
                    routed_reason="slm_http"
                )
                
                return {
                    "answer": response["content"],
                    "model_type": "slm",
                    "model_id": response["model_id"],
                    "topic": topic,
                    "routed_reason": "slm_http",
                    "training_job_enqueued": False
                }
                
            except HTTPSLMError as e:
                logger.warning("slm_http_failed", topic=topic, error=str(e))
                # Fall back to LLM
                return await self._route_to_llm(user_id, prompt, topic, request_id, "slm_unavailable")
        
        # Use local sLM as fallback
        slm = LocalSLM(topic=topic)
        slm.set_trained(True)  # Mark as trained since it's in registry
        
        # Generate response
        response = await slm.generate(prompt)
        
        # Store interaction
        await self._store_interaction(
            user_id=user_id,
            request_id=request_id,
            prompt=prompt,
            output=response["content"],
            topic=topic,
            model_type="slm",
            model_id=response["model_id"],
            latency_ms=response["latency_ms"],
            cost_usd=response["cost_usd"],
            routed_reason="slm_local"
        )
        
        return {
            "answer": response["content"],
            "model_type": "slm",
            "model_id": response["model_id"],
            "topic": topic,
            "routed_reason": "slm_local",
            "training_job_enqueued": False
        }
    
    async def _store_interaction(self, **kwargs):
        """Store interaction in database."""
        
        # Generate embedding for the prompt
        prompt_embedding = self.similarity_service.generate_embedding(kwargs["prompt"])
        output_hash = self.similarity_service.compute_prompt_hash(kwargs["output"])
        
        interaction = Interaction(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            topic=kwargs["topic"],
            prompt=kwargs["prompt"],
            output=kwargs["output"],
            model_type=kwargs["model_type"],
            model_id=kwargs["model_id"],
            latency_ms=kwargs["latency_ms"],
            cost_usd=kwargs["cost_usd"],
            routed_reason=kwargs["routed_reason"],
            prompt_embedding=self.similarity_service.embedding_to_json(prompt_embedding),
            output_hash=output_hash
        )
        
        self.db.add(interaction)
        self.db.commit()
        
        logger.info("interaction_stored", 
                   user_id=kwargs["user_id"], 
                   topic=kwargs["topic"],
                   model_type=kwargs["model_type"])
    
    async def _check_and_handle_repetition(self, 
                                         user_id: str, 
                                         prompt: str, 
                                         output: str, 
                                         topic: str) -> bool:
        """Check for repetition and enqueue training if needed."""
        
        # Generate embedding for current prompt
        current_embedding = self.similarity_service.generate_embedding(prompt)
        
        # Get recent LLM interactions for this topic
        recent_interactions = (
            self.db.query(Interaction)
            .filter(
                and_(
                    Interaction.topic == topic,
                    Interaction.model_type == "llm",
                    Interaction.prompt_embedding.isnot(None)
                )
            )
            .order_by(desc(Interaction.created_at))
            .limit(10)  # Check last 10 interactions
            .all()
        )
        
        if len(recent_interactions) < 2:
            return False  # Need at least 2 interactions to detect repetition
        
        # Check for similar prompts
        similar_count = 0
        for interaction in recent_interactions:
            if interaction.prompt_embedding:
                stored_embedding = self.similarity_service.embedding_from_json(
                    interaction.prompt_embedding
                )
                similarity = self.similarity_service.cosine_similarity(
                    current_embedding, stored_embedding
                )
                
                if similarity >= self.repetition_threshold:
                    similar_count += 1
        
        # Enqueue training if we detect repetition
        if similar_count >= 2:
            logger.info("repetition_detected", 
                       topic=topic, 
                       similar_count=similar_count,
                       threshold=self.repetition_threshold)
            
            job_id = await self.training_service.enqueue_training_job(topic)
            return job_id is not None
        
        return False
    
    def _find_ready_slm(self, topic: str) -> Optional[ModelRegistry]:
        """Find a ready sLM model for the given topic."""
        return (
            self.db.query(ModelRegistry)
            .filter(
                and_(
                    ModelRegistry.model_type == "slm",
                    ModelRegistry.topic == topic,
                    ModelRegistry.status == "ready"
                )
            )
            .order_by(desc(ModelRegistry.eval_score))
            .first()
        )