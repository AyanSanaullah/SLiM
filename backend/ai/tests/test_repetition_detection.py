"""Tests for repetition detection."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from src.ai_routing.workers.repetition_detector import RepetitionDetector
from src.ai_routing.core.embeddings import EmbeddingService
from src.ai_routing.models.interactions import Interaction
from src.ai_routing.models.training_jobs import TrainingJob, JobType, JobStatus


@pytest.fixture
def embedding_service():
    """Mock embedding service."""
    service = MagicMock(spec=EmbeddingService)
    service.generate_embedding.return_value = [0.1] * 384
    service.cosine_similarity.return_value = 0.95  # High similarity
    service.embedding_from_storage_format.return_value = [0.1] * 384
    return service


@pytest.fixture
def repetition_detector(embedding_service):
    """Create repetition detector."""
    return RepetitionDetector(embedding_service)


@pytest.mark.asyncio
async def test_detect_repetition_by_hash(repetition_detector, test_db):
    """Test repetition detection using output hash."""
    
    # Create similar interactions with same output hash
    base_time = datetime.utcnow()
    
    for i in range(3):
        interaction = Interaction(
            request_id=f"req_{i}",
            user_id="user1",
            topic="coding",
            prompt=f"Question {i}",
            output="Same response",
            model_type="llm",
            model_id=1,
            prompt_embedding='[0.1, 0.2, 0.3]',
            output_hash="same_hash_123",
            latency_ms=100,
            routed_reason="test",
            created_at=base_time - timedelta(minutes=i)
        )
        test_db.add(interaction)
    
    test_db.commit()
    
    # Check for repetition
    is_repetitive = await repetition_detector.check_repetition(
        topic="coding",
        prompt_embedding=[0.1] * 384,
        output_text="Same response",
        db=test_db
    )
    
    assert is_repetitive


@pytest.mark.asyncio
async def test_no_repetition_different_outputs(repetition_detector, test_db):
    """Test no repetition detected for different outputs."""
    
    base_time = datetime.utcnow()
    
    for i in range(3):
        interaction = Interaction(
            request_id=f"req_{i}",
            user_id="user1",
            topic="coding", 
            prompt=f"Question {i}",
            output=f"Different response {i}",
            model_type="llm",
            model_id=1,
            prompt_embedding='[0.1, 0.2, 0.3]',
            output_hash=f"hash_{i}",
            latency_ms=100,
            routed_reason="test",
            created_at=base_time - timedelta(minutes=i)
        )
        test_db.add(interaction)
    
    test_db.commit()
    
    # Mock low similarity
    repetition_detector.embedding_service.cosine_similarity.return_value = 0.3
    
    is_repetitive = await repetition_detector.check_repetition(
        topic="coding",
        prompt_embedding=[0.1] * 384,
        output_text="New unique response",
        db=test_db
    )
    
    assert not is_repetitive


@pytest.mark.asyncio
async def test_enqueue_training_job_finetune(repetition_detector, test_db):
    """Test enqueueing fine-tune job for new topic."""
    
    # Create enough interactions for training
    base_time = datetime.utcnow()
    
    for i in range(15):  # More than min_examples_for_training
        interaction = Interaction(
            request_id=f"req_{i}",
            user_id="user1",
            topic="new_topic",
            prompt=f"Question {i}",
            output=f"Response {i}",
            model_type="llm",
            model_id=1,
            prompt_embedding='[0.1, 0.2, 0.3]',
            output_hash=f"hash_{i}",
            latency_ms=100,
            routed_reason="test",
            created_at=base_time - timedelta(minutes=i)
        )
        test_db.add(interaction)
    
    test_db.commit()
    
    job_id = await repetition_detector.enqueue_training_job("new_topic", test_db)
    
    assert job_id is not None
    
    # Check job was created
    job = test_db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    assert job is not None
    assert job.job_type == JobType.FINETUNE
    assert job.status == JobStatus.PENDING
    assert job.topic == "new_topic"


@pytest.mark.asyncio
async def test_no_training_job_insufficient_data(repetition_detector, test_db):
    """Test no training job created with insufficient data."""
    
    # Create only a few interactions
    for i in range(3):
        interaction = Interaction(
            request_id=f"req_{i}",
            user_id="user1",
            topic="small_topic",
            prompt=f"Question {i}",
            output=f"Response {i}",
            model_type="llm",
            model_id=1,
            prompt_embedding='[0.1, 0.2, 0.3]',
            output_hash=f"hash_{i}",
            latency_ms=100,
            routed_reason="test"
        )
        test_db.add(interaction)
    
    test_db.commit()
    
    job_id = await repetition_detector.enqueue_training_job("small_topic", test_db)
    
    assert job_id is None