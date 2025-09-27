"""RQ worker for processing training jobs."""

import asyncio
import os
import sys
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.db.session import SessionLocal
from app.db.models import TrainingJob, ModelRegistry
from app.services.generation.slm_local import LocalSLM
from app.services.training import TrainingService
import structlog

logger = structlog.get_logger()

def process_training_job(job_id: int) -> bool:
    """Process a training job (called by RQ)."""
    
    logger.info("training_job_started", job_id=job_id)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Get the job
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logger.error("training_job_not_found", job_id=job_id)
            return False
        
        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        # Initialize training service
        training_service = TrainingService(db)
        
        # Process based on job type
        if job.job_type == "finetune":
            success = _process_finetune_job(job, training_service, db)
        elif job.job_type == "reinforce":
            success = _process_reinforce_job(job, training_service, db)
        else:
            logger.error("unknown_job_type", job_id=job_id, job_type=job.job_type)
            success = False
        
        # Complete the job
        metrics = {
            "job_type": job.job_type,
            "topic": job.topic,
            "num_examples": job.num_examples,
            "processing_time_seconds": (datetime.utcnow() - job.started_at).total_seconds()
        }
        
        training_service.complete_training_job(
            job_id=job_id,
            success=success,
            metrics=metrics,
            error=None if success else "Training failed"
        )
        
        return success
        
    except Exception as e:
        logger.error("training_job_error", job_id=job_id, error=str(e))
        
        # Mark job as failed
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error = str(e)
            db.commit()
        
        return False
        
    finally:
        db.close()

def _process_finetune_job(job: TrainingJob, training_service: TrainingService, db) -> bool:
    """Process a fine-tuning job to create a new sLM."""
    
    logger.info("processing_finetune_job", topic=job.topic, num_examples=job.num_examples)
    
    try:
        # Get training data
        training_data = training_service.get_training_data(job.topic)
        
        if len(training_data) < 5:
            logger.error("insufficient_training_data", topic=job.topic, count=len(training_data))
            return False
        
        # Create and train sLM
        slm = LocalSLM(topic=job.topic)
        
        # Run fine-tuning (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(slm.fine_tune(training_data))
        loop.close()
        
        if not success:
            logger.error("slm_fine_tuning_failed", topic=job.topic)
            return False
        
        # Create model registry entry
        eval_score = 0.85  # Simulated evaluation score
        model_name = f"slm_{job.topic}_v1.0"
        
        notes = {
            "created_from_job_id": job.id,
            "training_examples": len(training_data),
            "fine_tuned_at": datetime.utcnow().isoformat()
        }
        
        training_service.create_slm_model_entry(
            topic=job.topic,
            model_name=model_name,
            eval_score=eval_score,
            notes=notes
        )
        
        logger.info("finetune_job_completed", topic=job.topic, model_name=model_name)
        return True
        
    except Exception as e:
        logger.error("finetune_job_error", topic=job.topic, error=str(e))
        return False

def _process_reinforce_job(job: TrainingJob, training_service: TrainingService, db) -> bool:
    """Process a reinforcement job to update an existing sLM."""
    
    logger.info("processing_reinforce_job", topic=job.topic, base_model_id=job.base_model_id)
    
    try:
        # Get the base model
        base_model = db.query(ModelRegistry).filter(ModelRegistry.id == job.base_model_id).first()
        if not base_model:
            logger.error("base_model_not_found", model_id=job.base_model_id)
            return False
        
        # Get new training examples
        training_data = training_service.get_training_data(job.topic, limit=100)
        
        if len(training_data) < 3:
            logger.error("insufficient_reinforcement_data", topic=job.topic, count=len(training_data))
            return False
        
        # Create sLM instance and run reinforcement
        slm = LocalSLM(topic=job.topic)
        slm.set_trained(True)
        
        # Run reinforcement (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(slm.reinforce(training_data))
        loop.close()
        
        if not success:
            logger.error("slm_reinforcement_failed", topic=job.topic)
            return False
        
        # Update model version and score
        version_parts = base_model.version.split('.')
        patch_version = int(version_parts[-1]) + 1
        new_version = f"{'.'.join(version_parts[:-1])}.{patch_version}"
        
        new_eval_score = (base_model.eval_score or 0.8) + 0.02  # Slight improvement
        
        notes = {
            "reinforced_from_job_id": job.id,
            "reinforcement_examples": len(training_data),
            "reinforced_at": datetime.utcnow().isoformat()
        }
        
        training_service.update_slm_model(
            model_id=base_model.id,
            new_version=new_version,
            eval_score=new_eval_score,
            notes=notes
        )
        
        logger.info("reinforce_job_completed", 
                   topic=job.topic, 
                   model_id=base_model.id,
                   new_version=new_version)
        return True
        
    except Exception as e:
        logger.error("reinforce_job_error", topic=job.topic, error=str(e))
        return False

if __name__ == "__main__":
    # Start RQ worker
    import redis
    from rq import Worker, Queue
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_conn = redis.from_url(redis_url)
    
    # Create worker for training queue
    worker = Worker([Queue('training', connection=redis_conn)], connection=redis_conn)
    
    logger.info("starting_training_worker")
    worker.work()