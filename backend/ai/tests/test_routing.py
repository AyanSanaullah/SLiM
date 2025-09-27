"""Tests for the routing service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.ai_routing.core.routing import RoutingService
from src.ai_routing.core.topic_detection import TopicDetectionService
from src.ai_routing.core.embeddings import EmbeddingService
from src.ai_routing.models.model_registry import ModelRegistry, ModelType, ModelStatus
from src.ai_routing.providers.base import ModelResponse


@pytest.fixture
def embedding_service():
    """Mock embedding service."""
    service = MagicMock(spec=EmbeddingService)
    service.generate_embedding.return_value = [0.1] * 384  # Mock embedding
    return service


@pytest.fixture
def topic_detection_service(embedding_service):
    """Create topic detection service."""
    return TopicDetectionService(embedding_service)


@pytest.fixture
def routing_service(topic_detection_service):
    """Create routing service."""
    return RoutingService(topic_detection_service)


@pytest.mark.asyncio
async def test_route_to_default_llm(routing_service, sample_prompt, sample_user_id):
    """Test routing to default LLM when no sLM available."""
    
    # Mock provider
    mock_provider = AsyncMock()
    mock_provider.is_available.return_value = True
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-pro"
    
    # Mock provider factory
    routing_service.provider_factory.create_default_llm_provider = MagicMock(
        return_value=mock_provider
    )
    
    decision = await routing_service.route_prompt(
        prompt=sample_prompt,
        user_id=sample_user_id,
        topic_hint="coding"
    )
    
    assert decision.model_type == ModelType.LLM
    assert decision.topic == "coding"
    assert "default_llm" in decision.routed_reason


@pytest.mark.asyncio
async def test_route_to_slm_when_available(routing_service, test_db, sample_prompt, sample_user_id):
    """Test routing to sLM when available for topic."""
    
    # Create a ready sLM model in the database
    slm_model = ModelRegistry(
        model_type=ModelType.SLM,
        provider="local",
        model_name="coding_slm_v1",
        version="1.0",
        topic="coding",
        status=ModelStatus.READY,
        eval_score=0.85
    )
    test_db.add(slm_model)
    test_db.commit()
    
    # Mock provider
    mock_provider = AsyncMock()
    mock_provider.is_available.return_value = True
    
    routing_service.provider_factory.create_provider = MagicMock(
        return_value=mock_provider
    )
    
    decision = await routing_service.route_prompt(
        prompt=sample_prompt,
        user_id=sample_user_id,
        topic_hint="coding",
        db=test_db
    )
    
    assert decision.model_type == ModelType.SLM
    assert decision.topic == "coding"
    assert decision.model_registry.id == slm_model.id


@pytest.mark.asyncio
async def test_force_llm_flag(routing_service, test_db, sample_prompt, sample_user_id):
    """Test force_llm flag overrides sLM selection."""
    
    # Create a ready sLM model
    slm_model = ModelRegistry(
        model_type=ModelType.SLM,
        provider="local",
        model_name="coding_slm_v1",
        version="1.0",
        topic="coding",
        status=ModelStatus.READY
    )
    test_db.add(slm_model)
    test_db.commit()
    
    # Mock LLM provider
    mock_llm_provider = AsyncMock()
    mock_llm_provider.is_available.return_value = True
    mock_llm_provider.provider_name = "gemini"
    mock_llm_provider.model_name = "gemini-pro"
    
    routing_service.provider_factory.create_default_llm_provider = MagicMock(
        return_value=mock_llm_provider
    )
    
    decision = await routing_service.route_prompt(
        prompt=sample_prompt,
        user_id=sample_user_id,
        topic_hint="coding",
        force_llm=True,
        db=test_db
    )
    
    assert decision.model_type == ModelType.LLM
    assert "forced_llm_mode" in decision.routed_reason


def test_topic_detection_with_hint(topic_detection_service):
    """Test topic detection with explicit hint."""
    
    topic = topic_detection_service.detect_topic(
        prompt="Some prompt",
        topic_hint="mathematics"
    )
    
    assert topic == "math"  # Should be normalized


def test_topic_detection_with_keywords(topic_detection_service):
    """Test topic detection using keyword matching."""
    
    topic = topic_detection_service.detect_topic(
        prompt="How do I debug this Python function?"
    )
    
    assert topic == "coding"


def test_topic_detection_fallback(topic_detection_service):
    """Test topic detection fallback to general."""
    
    topic = topic_detection_service.detect_topic(
        prompt="Random question about nothing specific"
    )
    
    assert topic == "general"