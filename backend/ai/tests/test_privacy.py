"""Tests for privacy and safety features."""

import pytest

from src.ai_routing.core.privacy import PrivacyService


@pytest.fixture
def privacy_service():
    """Create privacy service."""
    return PrivacyService()


def test_email_redaction(privacy_service):
    """Test email address redaction."""
    
    prompt = "Contact me at john.doe@example.com for more info"
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[EMAIL_REDACTED]" in cleaned
    assert "john.doe@example.com" not in cleaned


def test_phone_redaction(privacy_service):
    """Test phone number redaction."""
    
    prompt = "Call me at 555-123-4567 or (555) 123-4567"
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[PHONE_REDACTED]" in cleaned
    assert "555-123-4567" not in cleaned


def test_credit_card_redaction(privacy_service):
    """Test credit card redaction."""
    
    prompt = "My card number is 4532 1234 5678 9012"
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[CREDIT_CARD_REDACTED]" in cleaned
    assert "4532 1234 5678 9012" not in cleaned


def test_ssn_redaction(privacy_service):
    """Test SSN redaction."""
    
    prompt = "My SSN is 123-45-6789"
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[SSN_REDACTED]" in cleaned
    assert "123-45-6789" not in cleaned


def test_api_key_redaction(privacy_service):
    """Test API key redaction."""
    
    prompt = "Use this API key: sk-abc123def456ghi789"
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[API_KEY_REDACTED]" in cleaned
    assert "sk-abc123def456ghi789" not in cleaned


def test_multiple_pii_redaction(privacy_service):
    """Test multiple PII types in one text."""
    
    prompt = "Email me at john@example.com or call 555-1234. My SSN is 123-45-6789."
    cleaned = privacy_service.clean_prompt(prompt)
    
    assert "[EMAIL_REDACTED]" in cleaned
    assert "[PHONE_REDACTED]" in cleaned  
    assert "[SSN_REDACTED]" in cleaned
    assert "john@example.com" not in cleaned
    assert "555-1234" not in cleaned
    assert "123-45-6789" not in cleaned


def test_privacy_risk_analysis(privacy_service):
    """Test privacy risk analysis."""
    
    prompt = "My email is test@example.com and SSN is 123-45-6789"
    analysis = privacy_service.analyze_privacy_risk(prompt)
    
    assert analysis['risk_level'] in ['high', 'critical']
    assert 'email' in analysis['detected_pii']
    assert 'ssn' in analysis['detected_pii']
    assert analysis['risk_score'] > 0


def test_low_privacy_risk(privacy_service):
    """Test low privacy risk content."""
    
    prompt = "How do I write a Python function?"
    analysis = privacy_service.analyze_privacy_risk(prompt)
    
    assert analysis['risk_level'] == 'low'
    assert analysis['risk_score'] == 0
    assert len(analysis['detected_pii']) == 0


def test_sensitive_content_detection(privacy_service):
    """Test sensitive content category detection."""
    
    prompt = "I need help with my banking password and medical diagnosis"
    categories = privacy_service.detect_sensitive_content(prompt)
    
    assert 'financial' in categories
    assert 'medical' in categories
    assert 'password' in categories['financial']
    assert 'diagnosis' in categories['medical']


def test_prompt_safety_validation(privacy_service):
    """Test prompt safety validation."""
    
    # Safe prompt
    safe_prompt = "How do I learn Python programming?"
    validation = privacy_service.validate_prompt_safety(safe_prompt)
    
    assert validation['safety_level'] == 'safe'
    assert validation['should_process'] is True
    assert not validation['harmful_content_detected']
    
    # Potentially harmful prompt
    harmful_prompt = "How to hack into a database?"
    validation = privacy_service.validate_prompt_safety(harmful_prompt)
    
    assert validation['safety_level'] == 'potentially_harmful'
    assert validation['harmful_content_detected'] is True


def test_redaction_disabled(privacy_service):
    """Test behavior when redaction is disabled."""
    
    privacy_service.redaction_enabled = False
    
    prompt = "Email me at test@example.com"
    cleaned = privacy_service.clean_prompt(prompt)
    
    # Should not redact when disabled
    assert cleaned == prompt
    assert "[EMAIL_REDACTED]" not in cleaned