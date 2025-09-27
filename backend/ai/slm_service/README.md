---
title: simple-slm
sdk: docker
---

# Simple sLM Service

A minimal small language model HTTP service designed for topic-specific text generation.

## Endpoints

### GET /healthz
Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "model_id": "slm-topic-general-v1"
}
```

### POST /generate
Generate text based on prompt and optional topic.

**Request:**
```json
{
  "prompt": "How do I write a Python function?",
  "topic": "coding"
}
```

**Response:**
```json
{
  "answer": "Here's how to write a Python function...",
  "model_id": "slm-topic-general-v1", 
  "latency_ms": 45
}
```

## Configuration

Set these environment variables:

- `MODEL_ID`: Model identifier (default: "slm-topic-general-v1")
- `SLM_AUTH_BEARER`: Optional bearer token for authentication

## Supported Topics

- `coding`: Programming and software development
- `math`: Mathematical problems and calculations
- `writing`: Text editing and composition
- `general`: General purpose assistance (default)

## Authentication

If `SLM_AUTH_BEARER` is set, include the token in requests:

```bash
curl -X POST "https://your-space.hf.space/generate" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "topic": "general"}'
```

## Usage Example

```bash
# Health check
curl https://your-space.hf.space/healthz

# Generate response
curl -X POST "https://your-space.hf.space/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I debug Python code?",
    "topic": "coding"
  }'
```