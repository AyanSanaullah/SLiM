# Development & Deployment Guide

## Quick Start - Local Development

1. **Clone and setup**:
   ```bash
   cd backend/ai
   cp .env.example .env
   # Edit .env with your Gemini API key
   ```

2. **Run with Docker**:
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:8000/v1/route \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "prompt": "What is AI?", "topic_hint": "technology"}'
   ```

## Hugging Face Spaces Deployment

### 1. Create Docker Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure:
   - **Name**: `your-username/slm-service`
   - **License**: Apache 2.0
   - **Space SDK**: Docker
   - **Visibility**: Public (for free tier)

### 2. Push Files to Space

Push these 4 files to your HF Space repository:

**`app.py`**:
```python
# Copy content from slm_service/app.py
```

**`requirements.txt`**:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
structlog==23.2.0
```

**`Dockerfile`**:
```dockerfile
# Copy content from slm_service/Dockerfile
```

**`README.md`**:
```markdown
---
title: sLM Service
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Small Language Model Service

HTTP service for routing AI prompts to specialized small models.
```

### 3. Configure Router for HTTP Mode

Update your router's `.env`:
```bash
# sLM Configuration
SLM_MODE=http
SLM_ENDPOINT_URL=https://your-username-slm-service.hf.space/generate
SLM_AUTH_BEARER=your-secret-token
SLM_TIMEOUT_MS=5000
```

### 4. Test Integration

```bash
# Test HF Space directly
curl -X POST https://your-username-slm-service.hf.space/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{"prompt": "Hello", "topic": "greeting"}'

# Test router with HTTP mode
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "prompt": "Hello world"}'
```

## Architecture

```
User Request → FastAPI Router → Topic Detection
                   ↓
          HTTP sLM Available? → YES → Hugging Face Space
                   ↓              ↓
                  NO          Success/Timeout
                   ↓              ↓
              Gemini LLM ← Fallback if sLM fails
                   ↓
            Store Interaction → Check Repetition → Queue Training
```

## Environment Variables

### Router Service
- `SLM_MODE`: `local` or `http`
- `SLM_ENDPOINT_URL`: HF Space URL + `/generate`
- `SLM_AUTH_BEARER`: Authentication token
- `SLM_TIMEOUT_MS`: Request timeout (default: 3000)
- `GEMINI_API_KEY`: Google Gemini API key

### sLM Service (HF Spaces)
- `AUTH_TOKEN`: Bearer token for authentication
- `LOG_LEVEL`: `INFO` or `DEBUG`

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/test_slm_http.py -v
```

## Monitoring

- Router metrics: `GET /metrics` (Prometheus format)
- Router health: `GET /health`
- sLM health: `GET /healthz`
- Logs: JSON structured logs with topic/latency tracking

## Cost Analysis

- **sLM (HTTP)**: Free on HF Spaces (rate limited)
- **LLM (Gemini)**: ~$0.001 per 1K tokens
- **Storage**: PostgreSQL for interactions
- **Cache**: Redis for embeddings

## Troubleshooting

1. **sLM timeout**: Increase `SLM_TIMEOUT_MS` or check HF Space logs
2. **Authentication fails**: Verify `SLM_AUTH_BEARER` matches Space token
3. **No fallback**: Router automatically falls back to Gemini on sLM failure
4. **Training not triggered**: Check repetition threshold (default: 0.90 similarity)