"""Observability utilities for metrics and logging."""

import os
import structlog
from prometheus_client import Counter, Histogram, Gauge
import re

# Setup structured logging
def setup_logging():
    """Configure structured logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Prometheus metrics
route_counter = Counter(
    'ai_routing_routes_total',
    'Total number of routing requests',
    ['model_type', 'topic', 'routed_reason']
)

route_latency = Histogram(
    'ai_routing_latency_seconds',
    'Request latency in seconds',
    ['model_type']
)

llm_requests = Counter(
    'ai_routing_llm_requests_total',
    'Total LLM requests'
)

slm_requests = Counter(
    'ai_routing_slm_requests_total',
    'Total sLM requests',
    ['topic']
)

repetition_detections = Counter(
    'ai_routing_repetitions_detected_total',
    'Total repetition detections',
    ['topic']
)

training_jobs = Counter(
    'ai_routing_training_jobs_total',
    'Total training jobs',
    ['job_type', 'status', 'topic']
)

model_cost = Counter(
    'ai_routing_cost_usd_total',
    'Total cost in USD',
    ['model_type']
)

active_slm_models = Gauge(
    'ai_routing_active_slm_models',
    'Number of active sLM models',
    ['topic']
)

errors = Counter(
    'ai_routing_errors_total',
    'Total errors',
    ['error_type']
)

class MetricsCollector:
    """Collector for application metrics."""
    
    def record_route(self, model_type: str, topic: str, latency_seconds: float, routed_reason: str = ""):
        """Record a routing decision."""
        route_counter.labels(
            model_type=model_type, 
            topic=topic, 
            routed_reason=routed_reason
        ).inc()
        
        route_latency.labels(model_type=model_type).observe(latency_seconds)
        
        if model_type == "llm":
            llm_requests.inc()
        else:
            slm_requests.labels(topic=topic).inc()
    
    def record_repetition(self, topic: str):
        """Record a repetition detection."""
        repetition_detections.labels(topic=topic).inc()
    
    def record_training_job(self, job_type: str, status: str, topic: str):
        """Record a training job event."""
        training_jobs.labels(job_type=job_type, status=status, topic=topic).inc()
    
    def record_cost(self, model_type: str, cost_usd: float):
        """Record model usage cost."""
        model_cost.labels(model_type=model_type).inc(cost_usd)
    
    def record_error(self, error_type: str):
        """Record an error."""
        errors.labels(error_type=error_type).inc()
    
    def update_active_models(self, topic_counts: dict):
        """Update active model counts."""
        for topic, count in topic_counts.items():
            active_slm_models.labels(topic=topic).set(count)

# Privacy utilities
class PrivacyFilter:
    """Simple PII redaction filter."""
    
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.store_raw = os.getenv("STORE_RAW_TEXT", "true").lower() == "true"
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing PII."""
        if self.store_raw:
            return text  # Store raw text if configured
        
        cleaned = self.email_pattern.sub('[EMAIL]', text)
        cleaned = self.phone_pattern.sub('[PHONE]', cleaned)
        return cleaned

# Global instances
_metrics = MetricsCollector()
_privacy_filter = PrivacyFilter()

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics

def get_privacy_filter() -> PrivacyFilter:
    """Get the global privacy filter."""
    return _privacy_filter