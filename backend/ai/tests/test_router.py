"""Tests for routing logic."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, ModelRegistry, Interaction
from app.services.router import RouterService
from app.services.similarity import SimilarityService

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def test_db():
    """Create test database."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def mock_gemini():
    """Mock Gemini LLM."""
    with patch('app.services.router.GeminiLLM') as mock:
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = {
            "content": "This is a mock LLM response.",
            "model_id": "gemini_test",
            "latency_ms": 100,
            "cost_usd": 0.001
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_route_to_llm_when_no_slm(test_db, mock_gemini):
    """Test routing to LLM when no sLM is available."""
    
    router = RouterService(test_db)
    
    result = await router.route_request(
        user_id="test_user",
        prompt="How do I write Python code?",
        topic_hint="coding",
        request_id="test_123"
    )
    
    assert result["model_type"] == "llm"
    assert result["topic"] == "coding"
    assert result["routed_reason"] == "no_slm_available"
    assert "This is a mock LLM response" in result["answer"]
    
    # Check interaction was stored
    interaction = test_db.query(Interaction).first()
    assert interaction is not None
    assert interaction.user_id == "test_user"
    assert interaction.topic == "coding"
    assert interaction.model_type == "llm"

@pytest.mark.asyncio
async def test_route_to_slm_when_available(test_db, mock_gemini):
    """Test routing to sLM when available."""
    
    # Create a ready sLM model
    slm_model = ModelRegistry(
        model_type="slm",
        provider="local",
        model_name="coding_slm",
        version="1.0",
        topic="coding",
        status="ready",
        eval_score=0.85
    )
    test_db.add(slm_model)
    test_db.commit()
    
    router = RouterService(test_db)
    
    result = await router.route_request(
        user_id="test_user",
        prompt="How do I debug Python?",
        topic_hint="coding",
        request_id="test_124"
    )
    
    assert result["model_type"] == "slm"
    assert result["topic"] == "coding"
    assert result["routed_reason"] == "slm_available"
    
    # Should not have called Gemini
    mock_gemini.generate.assert_not_called()

@pytest.mark.asyncio
async def test_force_llm_override(test_db, mock_gemini):
    """Test force_llm flag overrides sLM selection."""
    
    # Create a ready sLM model
    slm_model = ModelRegistry(
        model_type="slm",
        provider="local", 
        model_name="coding_slm",
        version="1.0",
        topic="coding",
        status="ready"
    )
    test_db.add(slm_model)
    test_db.commit()
    
    router = RouterService(test_db)
    
    result = await router.route_request(
        user_id="test_user",
        prompt="How do I code?",
        topic_hint="coding",
        force_llm=True,
        request_id="test_125"
    )
    
    assert result["model_type"] == "llm"
    assert result["routed_reason"] == "forced_llm"
    
    # Should have called Gemini despite sLM being available
    mock_gemini.generate.assert_called_once()

@pytest.mark.asyncio
async def test_repetition_detection(test_db, mock_gemini):
    """Test repetition detection triggers training."""
    
    router = RouterService(test_db)
    
    # Create similar interactions
    base_prompt = "How do I write a for loop in Python?"
    similar_prompts = [
        "How do I write a for loop in Python?",
        "How do I create a for loop in Python?", 
        "How to write for loops in Python?"
    ]
    
    with patch.object(router.training_service, 'enqueue_training_job', return_value=1) as mock_enqueue:
        # Submit similar prompts
        for i, prompt in enumerate(similar_prompts):
            result = await router.route_request(
                user_id="test_user",
                prompt=prompt,
                topic_hint="coding",
                request_id=f"test_rep_{i}"
            )
        
        # Last request should detect repetition
        assert result["training_job_enqueued"] == True
        mock_enqueue.assert_called_once_with("coding")

def test_topic_detection():
    """Test topic detection logic."""
    
    similarity_service = SimilarityService()
    
    # Test keyword-based detection
    coding_topic = similarity_service.detect_topic("How do I debug my Python code?")
    assert coding_topic == "coding"
    
    math_topic = similarity_service.detect_topic("Solve this algebra equation for x")
    assert math_topic == "math"
    
    # Test hint override
    hint_topic = similarity_service.detect_topic("Random text", topic_hint="writing")
    assert hint_topic == "writing"
    
    # Test fallback
    general_topic = similarity_service.detect_topic("Random unrelated question")
    assert general_topic == "general"

def test_similarity_calculation():
    """Test embedding similarity calculation."""
    
    similarity_service = SimilarityService()
    
    # Test identical strings
    emb1 = similarity_service.generate_embedding("Hello world")
    emb2 = similarity_service.generate_embedding("Hello world")
    similarity = similarity_service.cosine_similarity(emb1, emb2)
    assert similarity > 0.99
    
    # Test similar strings
    emb3 = similarity_service.generate_embedding("How to code in Python")
    emb4 = similarity_service.generate_embedding("How to program in Python")
    similarity2 = similarity_service.cosine_similarity(emb3, emb4)
    assert similarity2 > 0.8
    
    # Test different strings
    emb5 = similarity_service.generate_embedding("Python programming")
    emb6 = similarity_service.generate_embedding("Cooking recipes")
    similarity3 = similarity_service.cosine_similarity(emb5, emb6)
    assert similarity3 < 0.5