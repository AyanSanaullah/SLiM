"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from src.ai_routing.models.base import Base
from src.ai_routing.core.database import get_db
from src.ai_routing.api.app import create_app
from src.ai_routing.config import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings."""
    return Settings(
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/1",  # Use different DB for tests
        log_level="DEBUG",
        metrics_enabled=False,
        force_llm=False,
        force_slm=False,
        pii_redaction_enabled=True,
        store_raw_prompts=True
    )


@pytest.fixture(scope="session")
def test_engine(test_settings):
    """Create test database engine."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False}  # For SQLite
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db(test_engine) -> Generator[Session, None, None]:
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_app(test_db):
    """Create test FastAPI app."""
    app = create_app()
    
    # Override dependency
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_prompt():
    """Sample user prompt for testing."""
    return "How do I implement a binary search algorithm in Python?"


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "test_user_123"