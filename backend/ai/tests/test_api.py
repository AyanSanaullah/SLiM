"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch

from src.ai_routing.api.models import RouteRequest


def test_health_endpoint(client):
    """Test health check endpoint."""
    
    with patch('src.ai_routing.workers.cache_service.CacheService.health_check', new_callable=AsyncMock) as mock_cache:
        mock_cache.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data


def test_stats_endpoint(client, test_db):
    """Test statistics endpoint."""
    
    response = client.get("/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert "routing_stats" in data
    assert "topic_stats" in data
    assert "total_interactions" in data


@pytest.mark.asyncio
async def test_route_endpoint_success(client, sample_prompt, sample_user_id):
    """Test successful routing request."""
    
    request_data = {
        "user_id": sample_user_id,
        "prompt": sample_prompt,
        "topic_hint": "coding",
        "force_llm": False
    }
    
    with patch('src.ai_routing.core.routing.RoutingService.route_prompt', new_callable=AsyncMock) as mock_route, \
         patch('src.ai_routing.workers.cache_service.CacheService.get_cached_response', new_callable=AsyncMock) as mock_cache, \
         patch('src.ai_routing.workers.repetition_detector.RepetitionDetector.check_repetition', new_callable=AsyncMock) as mock_repetition:
        
        # Mock cache miss
        mock_cache.return_value = None
        
        # Mock routing decision
        from src.ai_routing.core.routing import RoutingDecision
        from src.ai_routing.models.model_registry import ModelRegistry, ModelType, ModelStatus
        from src.ai_routing.providers.base import ModelResponse
        
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = ModelResponse(
            content="Here's how to implement binary search...",
            model_id="test_model",
            latency_ms=150,
            cost_usd=0.001
        )
        
        mock_model = ModelRegistry(
            id=1,
            model_type=ModelType.LLM,
            provider="gemini",
            model_name="gemini-pro",
            version="1.0",
            topic="coding",
            status=ModelStatus.READY
        )
        
        mock_decision = RoutingDecision(
            model_type=ModelType.LLM,
            model_registry=mock_model,
            provider=mock_provider,
            routed_reason="no_suitable_slm_available",
            topic="coding"
        )
        
        mock_route.return_value = mock_decision
        mock_repetition.return_value = False
        
        response = client.post("/v1/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_type"] == "llm"
        assert data["topic"] == "coding"
        assert "answer" in data
        assert "latency_ms" in data


def test_route_endpoint_validation_error(client):
    """Test routing request with validation errors."""
    
    # Missing required fields
    request_data = {
        "prompt": "Some prompt"
        # Missing user_id
    }
    
    response = client.post("/v1/route", json=request_data)
    
    assert response.status_code == 422  # Validation error


def test_route_endpoint_with_request_id(client, sample_prompt, sample_user_id):
    """Test routing request with custom request ID."""
    
    request_data = {
        "user_id": sample_user_id,
        "prompt": sample_prompt,
        "request_id": "custom_req_123"
    }
    
    with patch('src.ai_routing.core.routing.RoutingService.route_prompt', new_callable=AsyncMock) as mock_route, \
         patch('src.ai_routing.workers.cache_service.CacheService.get_cached_response', new_callable=AsyncMock) as mock_cache:
        
        # Setup mocks similar to previous test
        mock_cache.return_value = None
        
        from src.ai_routing.core.routing import RoutingDecision
        from src.ai_routing.models.model_registry import ModelRegistry, ModelType, ModelStatus
        from src.ai_routing.providers.base import ModelResponse
        
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = ModelResponse(
            content="Response",
            model_id="test_model",
            latency_ms=100,
            cost_usd=0.0
        )
        
        mock_model = ModelRegistry(
            id=1,
            model_type=ModelType.SLM,
            provider="local",
            model_name="local_slm",
            version="1.0", 
            topic="general",
            status=ModelStatus.READY
        )
        
        mock_decision = RoutingDecision(
            model_type=ModelType.SLM,
            model_registry=mock_model,
            provider=mock_provider,
            routed_reason="default_slm",
            topic="general"
        )
        
        mock_route.return_value = mock_decision
        
        response = client.post("/v1/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "custom_req_123"


def test_route_request_model_validation():
    """Test RouteRequest model validation."""
    
    # Valid request
    valid_request = RouteRequest(
        user_id="user123",
        prompt="Test prompt",
        topic_hint="coding",
        force_llm=True,
        request_id="req123"
    )
    
    assert valid_request.user_id == "user123"
    assert valid_request.prompt == "Test prompt"
    assert valid_request.topic_hint == "coding"
    assert valid_request.force_llm is True
    assert valid_request.request_id == "req123"
    
    # Test defaults
    minimal_request = RouteRequest(
        user_id="user123",
        prompt="Test prompt"
    )
    
    assert minimal_request.topic_hint is None
    assert minimal_request.force_llm is False
    assert minimal_request.request_id is None