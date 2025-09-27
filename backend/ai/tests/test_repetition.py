"""Tests for repetition detection logic."""

import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Interaction, TrainingJob
from app.services.similarity import SimilarityService
from app.services.training import TrainingService

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_repetition.db"
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
def similarity_service():
    """Create similarity service."""
    return SimilarityService()

def test_embedding_generation(similarity_service):
    """Test that embeddings are generated correctly."""
    
    text = "How do I write Python code?"
    embedding = similarity_service.generate_embedding(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

def test_cosine_similarity_identical(similarity_service):
    """Test cosine similarity for identical texts."""
    
    text = "How to learn programming"
    emb1 = similarity_service.generate_embedding(text)
    emb2 = similarity_service.generate_embedding(text)
    
    similarity = similarity_service.cosine_similarity(emb1, emb2)
    assert similarity > 0.99

def test_cosine_similarity_similar(similarity_service):
    """Test cosine similarity for similar texts."""
    
    text1 = "How to learn Python programming"
    text2 = "How to study Python coding"
    
    emb1 = similarity_service.generate_embedding(text1)
    emb2 = similarity_service.generate_embedding(text2)
    
    similarity = similarity_service.cosine_similarity(emb1, emb2)
    assert similarity > 0.7  # Should be quite similar

def test_cosine_similarity_different(similarity_service):
    """Test cosine similarity for different texts."""
    
    text1 = "Python programming tutorial"
    text2 = "Cooking chocolate cake recipe"
    
    emb1 = similarity_service.generate_embedding(text1)
    emb2 = similarity_service.generate_embedding(text2)
    
    similarity = similarity_service.cosine_similarity(emb1, emb2)
    assert similarity < 0.5  # Should be quite different

def test_find_similar_prompts(similarity_service):
    """Test finding similar prompts above threshold."""
    
    target_text = "How to debug Python code"
    similar_texts = [
        "How to debug Python programs",
        "Debugging Python applications",
        "Cooking pasta recipes",  # Different topic
        "Python debugging techniques"
    ]
    
    target_emb = similarity_service.generate_embedding(target_text)
    candidate_embs = [similarity_service.generate_embedding(text) for text in similar_texts]
    
    similar_indices = similarity_service.find_similar_prompts(
        target_emb, candidate_embs, threshold=0.7
    )
    
    # Should find the Python-related texts but not cooking
    assert len(similar_indices) >= 2
    assert 2 not in similar_indices  # Cooking recipe should not be similar

def test_topic_detection_keywords(similarity_service):
    """Test topic detection using keywords."""
    
    # Test coding detection
    coding_prompts = [
        "How do I write a Python function?",
        "Debug this JavaScript code",
        "SQL query optimization"
    ]
    
    for prompt in coding_prompts:
        topic = similarity_service.detect_topic(prompt)
        assert topic == "coding"
    
    # Test math detection
    math_prompts = [
        "Solve this algebra equation",
        "Calculate the derivative",
        "Mathematics problem solving"
    ]
    
    for prompt in math_prompts:
        topic = similarity_service.detect_topic(prompt)
        assert topic == "math"

def test_topic_detection_with_hint(similarity_service):
    """Test topic detection with explicit hint."""
    
    # Hint should override keyword detection
    topic = similarity_service.detect_topic(
        "Some random text about anything", 
        topic_hint="coding"
    )
    assert topic == "coding"
    
    # Test hint normalization
    topic = similarity_service.detect_topic(
        "Text", 
        topic_hint="programming"
    )
    assert topic == "coding"  # Should map programming -> coding

@pytest.mark.asyncio
async def test_training_job_enqueue(test_db):
    """Test training job creation logic."""
    
    with patch('redis.from_url'), patch('rq.Queue') as mock_queue:
        mock_queue.return_value.enqueue = MagicMock()
        
        training_service = TrainingService(test_db)
        
        # Create enough interactions for training
        for i in range(6):  # More than min_examples (5)
            interaction = Interaction(
                request_id=f"req_{i}",
                user_id="test_user",
                topic="coding",
                prompt=f"How to code example {i}",
                output=f"Response {i}",
                model_type="llm",
                model_id="gemini_test",
                latency_ms=100,
                cost_usd=0.001,
                routed_reason="test",
                prompt_embedding="[]",
                output_hash=f"hash_{i}"
            )
            test_db.add(interaction)
        
        test_db.commit()
        
        # Enqueue training job
        job_id = await training_service.enqueue_training_job("coding")
        
        assert job_id is not None
        
        # Check job was created
        job = test_db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        assert job is not None
        assert job.topic == "coding"
        assert job.job_type == "finetune"  # No existing sLM
        assert job.status == "pending"
        assert job.num_examples == 6

@pytest.mark.asyncio
async def test_insufficient_examples_no_job(test_db):
    """Test that no job is created with insufficient examples."""
    
    with patch('redis.from_url'), patch('rq.Queue'):
        training_service = TrainingService(test_db)
        
        # Create only 2 interactions (less than min_examples)
        for i in range(2):
            interaction = Interaction(
                request_id=f"req_{i}",
                user_id="test_user", 
                topic="coding",
                prompt=f"Prompt {i}",
                output=f"Response {i}",
                model_type="llm",
                model_id="gemini_test",
                latency_ms=100,
                cost_usd=0.001,
                routed_reason="test",
                prompt_embedding="[]",
                output_hash=f"hash_{i}"
            )
            test_db.add(interaction)
        
        test_db.commit()
        
        # Should not enqueue job
        job_id = await training_service.enqueue_training_job("coding")
        assert job_id is None

def test_prompt_hash_consistency(similarity_service):
    """Test that prompt hashing is consistent."""
    
    prompt = "How do I learn Python?"
    
    hash1 = similarity_service.compute_prompt_hash(prompt)
    hash2 = similarity_service.compute_prompt_hash(prompt)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex length
    
    # Different prompts should have different hashes
    hash3 = similarity_service.compute_prompt_hash("Different prompt")
    assert hash1 != hash3

def test_embedding_json_serialization(similarity_service):
    """Test embedding serialization to/from JSON."""
    
    text = "Test embedding serialization"
    embedding = similarity_service.generate_embedding(text)
    
    # Serialize to JSON
    json_str = similarity_service.embedding_to_json(embedding)
    assert isinstance(json_str, str)
    
    # Deserialize from JSON
    restored_embedding = similarity_service.embedding_from_json(json_str)
    assert restored_embedding == embedding

def test_topic_centroids_update(similarity_service):
    """Test topic centroid updates."""
    
    # Add examples for a topic
    coding_examples = [
        "How to write Python code",
        "JavaScript programming tutorial", 
        "SQL database queries"
    ]
    
    similarity_service.add_topic_examples("coding", coding_examples)
    
    # Check that centroid was created
    topics = similarity_service.get_available_topics()
    assert "coding" in topics
    
    # Test that new prompts can be matched to this topic
    test_prompt = "How to debug Python programs"
    test_embedding = similarity_service.generate_embedding(test_prompt)
    
    best_topic, similarity = similarity_service._find_best_topic_by_embedding(test_embedding)
    assert best_topic == "coding"
    assert similarity > 0.5