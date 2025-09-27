# AI Routing Service

A minimal, production-ready service that intelligently routes prompts between small language models (sLM) and large language models (LLM), with automatic training based on repetition detection.

## Architecture

- **FastAPI** - Web framework with async support
- **PostgreSQL** - Persistent storage for interactions and models  
- **Redis** - Caching and job queue
- **RQ** - Background job processing
- **Sentence Transformers** - Local embeddings for topic detection
- **Gemini API** - LLM provider

## Core Features

1. **Intelligent Routing**: Routes to sLM if available for topic, otherwise to LLM
2. **Topic Detection**: Uses embeddings + keywords to classify prompts
3. **Repetition Detection**: Monitors LLM responses for similar patterns
4. **Automatic Training**: Creates/reinforces sLM models when repetition detected
5. **Privacy**: Simple PII redaction before database storage
6. **Observability**: Prometheus metrics and structured logging

## Quick Start

1. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

2. **Start services**:
```bash
docker compose -f docker-compose.dev.yml up --build
```

3. **Test the API**:
```bash
curl -X POST "http://localhost:8000/v1/route" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "How do I write a Python function?",
    "topic_hint": "coding"
  }'
```

## API Reference

### POST /v1/route

Route a prompt to the appropriate model.

**Request**:
```json
{
  "user_id": "string",
  "prompt": "string", 
  "topic_hint": "string (optional)",
  "force_llm": "boolean (optional)",
  "request_id": "string (optional)"
}
```

**Response**:
```json
{
  "answer": "string",
  "model_type": "slm|llm",
  "model_id": "string",
  "topic": "string", 
  "routed_reason": "string",
  "training_job_enqueued": "boolean"
}
```

### GET /health

Health check endpoint.

### GET /metrics

Prometheus metrics endpoint.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `ROUTER_FORCE_LLM` | Always route to LLM | `false` |
| `ROUTER_DISABLE_SLM` | Never route to sLM | `false` |
| `REPETITION_SIM_THRESHOLD` | Similarity threshold for repetition | `0.90` |
| `STORE_RAW_TEXT` | Store raw prompts (vs redacted) | `true` |

## Workflow

1. **First Request**: No sLM exists → routes to LLM → stores interaction
2. **Similar Requests**: Embedding similarity ≥ threshold → enqueues training job
3. **Training Completes**: New sLM created and registered as "ready"
4. **Subsequent Requests**: Routes to sLM for faster, cheaper responses

## Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

Key test scenarios:
- ✅ Routes to LLM when no sLM available
- ✅ Routes to sLM when ready model exists  
- ✅ Respects force_llm flag
- ✅ Detects repetition above threshold
- ✅ Enqueues training jobs correctly
- ✅ Topic detection works with keywords and hints

## Metrics

Available at `/metrics`:
- `ai_routing_routes_total` - Total routes by model type
- `ai_routing_latency_seconds` - Response latency
- `ai_routing_repetitions_detected_total` - Repetition detections
- `ai_routing_training_jobs_total` - Training job counts
- `ai_routing_cost_usd_total` - Total API costs

## Database Schema

**interactions**: User prompts and model responses
- `id`, `user_id`, `prompt`, `output`, `model_type`, `topic`, `created_at`, etc.

**model_registry**: Available models and their status  
- `id`, `model_type`, `topic`, `status`, `eval_score`, `version`, etc.

**training_jobs**: Background training job tracking
- `id`, `topic`, `job_type`, `status`, `num_examples`, `metrics`, etc.

## Acceptance Criteria

✅ **No Google agent dependencies** - Uses only Gemini API  
✅ **Docker Compose startup** - Single command deployment  
✅ **Routes to LLM initially** - When no sLM exists  
✅ **Repetition detection** - Triggers training jobs  
✅ **Training pipeline** - Creates ready sLM models  
✅ **Switches to sLM** - After training completes  
✅ **Correct response format** - All required fields present