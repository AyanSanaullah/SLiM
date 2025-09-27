"""Tests for HTTP sLM adapter."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.generation.slm_http import generate_slm, HTTPSLMError, check_slm_health, is_http_slm_enabled


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict('os.environ', {
        'SLM_ENDPOINT_URL': 'https://test-slm.hf.space/generate',
        'SLM_AUTH_BEARER': 'test-token',
        'SLM_TIMEOUT_MS': '2000',
        'SLM_MODE': 'http'
    }):
        yield


@pytest.mark.asyncio
async def test_generate_slm_success(mock_env_vars):
    """Test successful HTTP sLM generation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "Test response from sLM",
        "model_id": "test-slm-v1",
        "latency_ms": 150
    }
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await generate_slm("What is the weather?", "weather")
        
        assert result["content"] == "Test response from sLM"
        assert result["model_id"] == "test-slm-v1"
        assert result["cost_usd"] == 0.0
        assert result["latency_ms"] > 0
        assert result["metadata"]["endpoint"] == "https://test-slm.hf.space/generate"
        assert result["metadata"]["topic"] == "weather"


@pytest.mark.asyncio
async def test_generate_slm_timeout(mock_env_vars):
    """Test HTTP sLM timeout with fallback to LLM."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )
        
        with pytest.raises(HTTPSLMError) as exc_info:
            await generate_slm("What is AI?", "technology")
        
        assert "Request timeout" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_slm_connection_error(mock_env_vars):
    """Test HTTP sLM connection error."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        
        with pytest.raises(HTTPSLMError) as exc_info:
            await generate_slm("Test prompt", "general")
        
        assert "Connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_slm_auth_error(mock_env_vars):
    """Test HTTP sLM authentication error."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(HTTPSLMError) as exc_info:
            await generate_slm("Test prompt", "general")
        
        assert "Authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_slm_server_error(mock_env_vars):
    """Test HTTP sLM server error."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(HTTPSLMError) as exc_info:
            await generate_slm("Test prompt", "general")
        
        assert "Server error: 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_slm_no_endpoint():
    """Test HTTP sLM with no endpoint configured."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(HTTPSLMError) as exc_info:
            await generate_slm("Test prompt", "general")
        
        assert "SLM_ENDPOINT_URL not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_slm_health_success(mock_env_vars):
    """Test successful health check."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        result = await check_slm_health()
        
        assert result is True


@pytest.mark.asyncio
async def test_check_slm_health_failure(mock_env_vars):
    """Test failed health check."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        result = await check_slm_health()
        
        assert result is False


@pytest.mark.asyncio
async def test_check_slm_health_no_endpoint():
    """Test health check with no endpoint."""
    with patch.dict('os.environ', {}, clear=True):
        result = await check_slm_health()
        assert result is False


def test_is_http_slm_enabled():
    """Test HTTP sLM mode detection."""
    with patch.dict('os.environ', {'SLM_MODE': 'http'}):
        assert is_http_slm_enabled() is True
    
    with patch.dict('os.environ', {'SLM_MODE': 'local'}):
        assert is_http_slm_enabled() is False
    
    with patch.dict('os.environ', {}, clear=True):
        assert is_http_slm_enabled() is False


@pytest.mark.asyncio
async def test_generate_slm_request_headers(mock_env_vars):
    """Test that proper headers are sent with request."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "Test response",
        "model_id": "test-model"
    }
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post
        
        await generate_slm("Test prompt", "general")
        
        # Check that post was called with correct headers
        call_args = mock_post.call_args
        headers = call_args.kwargs['headers']
        
        assert headers['Content-Type'] == 'application/json'
        assert headers['Authorization'] == 'Bearer test-token'


@pytest.mark.asyncio
async def test_generate_slm_request_payload(mock_env_vars):
    """Test that proper payload is sent with request."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "Test response",
        "model_id": "test-model"
    }
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post
        
        await generate_slm("What is AI?", "technology")
        
        # Check that post was called with correct payload
        call_args = mock_post.call_args
        payload = call_args.kwargs['json']
        
        assert payload['prompt'] == 'What is AI?'
        assert payload['topic'] == 'technology'