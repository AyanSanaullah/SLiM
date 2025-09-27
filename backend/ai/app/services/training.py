"""Training service for managing sLM training jobs."""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import redis
from rq import Queue
import structlog

from ..db.models import TrainingJob, ModelRegistry, Interaction

logger = structlog.get_logger()

class TrainingService:
    """Service for managing training jobs and model updates."""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Setup Redis and RQ
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_conn = redis.from_url(redis_url)
        self.training_queue = Queue('training', connection=self.redis_conn)
        
        # Configuration
        self.min_examples = int(os.getenv("MIN_EXAMPLES_FOR_TRAINING", "5"))
    
    async def enqueue_training_job(self, topic: str) -> Optional[int]:
        """Enqueue a training job for the given topic."""
        
        # Check if we have enough examples
        example_count = (
            self.db.query(Interaction)
            .filter(
                and_(
                    Interaction.topic == topic,
                    Interaction.model_type == "llm"
                )
            )
            .count()
        )
        
        if example_count < self.min_examples:
            logger.info("insufficient_examples_for_training", 
                       topic=topic, 
                       count=example_count,
                       required=self.min_examples)
            return None
        
        # Check if there's already a pending/running job
        existing_job = (
            self.db.query(TrainingJob)
            .filter(
                and_(
                    TrainingJob.topic == topic,
                    TrainingJob.status.in_(["pending", "running"])
                )
            )
            .first()
        )
        
        if existing_job:
            logger.info("training_job_already_exists", topic=topic, job_id=existing_job.id)
            return existing_job.id
        
        # Determine job type
        existing_slm = (
            self.db.query(ModelRegistry)
            .filter(
                and_(
                    ModelRegistry.model_type == "slm",
                    ModelRegistry.topic == topic,
                    ModelRegistry.status == "ready"
                )
            )
            .first()
        )
        
        job_type = "reinforce" if existing_slm else "finetune"
        base_model_id = existing_slm.id if existing_slm else None
        
        # Create training job
        training_job = TrainingJob(
            topic=topic,
            base_model_id=base_model_id,
            job_type=job_type,
            status="pending",
            num_examples=example_count
        )
        
        self.db.add(training_job)
        self.db.commit()
        
        # Enqueue the job
        self.training_queue.enqueue(
            'workers.worker.process_training_job',
            training_job.id,
            job_timeout=1800  # 30 minutes
        )
        
        logger.info("training_job_enqueued", 
                   job_id=training_job.id,
                   topic=topic,
                   job_type=job_type,
                   num_examples=example_count)
        
        return training_job.id
    
    def get_training_data(self, topic: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get training data for a topic."""
        
        interactions = (
            self.db.query(Interaction)
            .filter(
                and_(
                    Interaction.topic == topic,
                    Interaction.model_type == "llm"
                )
            )
            .order_by(desc(Interaction.created_at))
            .limit(limit)
            .all()
        )
        
        return [
            {
                "prompt": interaction.prompt,
                "output": interaction.output,
                "user_id": interaction.user_id,
                "created_at": interaction.created_at.isoformat()
            }
            for interaction in interactions
        ]
    
    def create_slm_model_entry(self, 
                              topic: str, 
                              model_name: str,
                              eval_score: float = None,
                              notes: Dict[str, Any] = None) -> ModelRegistry:
        """Create a new sLM model entry in the registry."""
        
        model = ModelRegistry(
            model_type="slm",
            provider="local",
            model_name=model_name,
            version="1.0",
            topic=topic,
            status="ready",
            eval_score=eval_score,
            notes=notes or {}
        )
        
        self.db.add(model)
        self.db.commit()
        
        logger.info("slm_model_created", 
                   model_id=model.id,
                   topic=topic,
                   model_name=model_name)
        
        return model
    
    def update_slm_model(self, 
                        model_id: int,
                        new_version: str = None,
                        eval_score: float = None,
                        notes: Dict[str, Any] = None) -> bool:
        """Update an existing sLM model."""
        
        model = self.db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        if not model:
            return False
        
        if new_version:
            model.version = new_version
        if eval_score is not None:
            model.eval_score = eval_score
        if notes:
            model.notes.update(notes)
        
        self.db.commit()
        
        logger.info("slm_model_updated", 
                   model_id=model_id,
                   new_version=new_version,
                   eval_score=eval_score)
        
        return True
    
    def complete_training_job(self, 
                             job_id: int, 
                             success: bool,
                             metrics: Dict[str, Any] = None,
                             error: str = None) -> bool:
        """Mark a training job as completed."""
        
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return False
        
        job.status = "completed" if success else "failed"
        job.completed_at = datetime.utcnow()
        
        if metrics:
            job.metrics = metrics
        if error:
            job.error = error
        
        self.db.commit()
        
        logger.info("training_job_completed", 
                   job_id=job_id,
                   success=success,
                   topic=job.topic)
        
        return True
    
    def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a training job."""
        
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return None
        
        return {
            "id": job.id,
            "topic": job.topic,
            "job_type": job.job_type,
            "status": job.status,
            "num_examples": job.num_examples,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metrics": job.metrics,
            "error": job.error
        }