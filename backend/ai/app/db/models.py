"""Database models for the AI routing service."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func

Base = declarative_base()


class Interaction(Base):
    """Store all user interactions and model responses."""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True)
    request_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    topic = Column(String(50), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    output = Column(Text, nullable=False)
    model_type = Column(String(10), nullable=False)  # 'slm' or 'llm'
    model_id = Column(String(100), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    cost_usd = Column(Float, default=0.0)
    routed_reason = Column(String(200), nullable=False)
    prompt_embedding = Column(JSON)  # Store as JSON array
    output_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class ModelRegistry(Base):
    """Registry of available models."""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(10), nullable=False)  # 'slm' or 'llm'
    provider = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    topic = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)  # 'ready', 'training', 'deprecated'
    eval_score = Column(Float)
    notes = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class TrainingJob(Base):
    """Training jobs for model creation and reinforcement."""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True)
    topic = Column(String(50), nullable=False, index=True)
    base_model_id = Column(Integer)  # Reference to model_registry.id for reinforcement
    job_type = Column(String(20), nullable=False)  # 'reinforce' or 'finetune'
    status = Column(String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed'
    num_examples = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    metrics = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)