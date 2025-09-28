"""
String Comparison Service Client
Client for integrating with the string comparison service during training cycles
"""

import requests
import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class StringComparisonClient:
    """
    Client for the string comparison service at http://0.0.0.0:8000/
    """
    
    def __init__(self, base_url: str = "http://0.0.0.0:8000"):
        self.base_url = base_url
        self.compare_endpoint = f"{base_url}/compare"
        self.health_endpoint = f"{base_url}/health"
        
    def is_service_available(self) -> bool:
        """
        Check if the string comparison service is available
        
        Returns:
            bool: True if service is available, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"String comparison service not available: {e}")
            return False
    
    def compare_sentences(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """
        Compare two sentences using the string comparison service
        
        Args:
            sentence1: First sentence (model response)
            sentence2: Second sentence (expected answer)
            
        Returns:
            Dictionary with comparison results or error information
        """
        try:
            # Prepare request payload
            payload = {
                "sentence1": sentence1,
                "sentence2": sentence2
            }
            
            # Make request to comparison service
            response = requests.post(
                self.compare_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract similarity score and determine quality
                similarity_score = result.get('similarity', 0.0)
                quality_label = self._determine_quality_label(similarity_score)
                
                return {
                    'success': True,
                    'similarity_score': similarity_score,
                    'similarity_percentage': f"{similarity_score * 100:.2f}%",
                    'quality_label': quality_label,
                    'sentence1': result.get('sentence1', sentence1),
                    'sentence2': result.get('sentence2', sentence2),
                    'raw_response': result
                }
            else:
                logger.error(f"String comparison service error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'similarity_score': 0.0,
                    'quality_label': 'ERROR'
                }
                
        except requests.exceptions.Timeout:
            logger.error("String comparison service timeout")
            return {
                'success': False,
                'error': 'Service timeout',
                'similarity_score': 0.0,
                'quality_label': 'ERROR'
            }
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to string comparison service")
            return {
                'success': False,
                'error': 'Connection error - service may be offline',
                'similarity_score': 0.0,
                'quality_label': 'ERROR'
            }
        except Exception as e:
            logger.error(f"Unexpected error in string comparison: {e}")
            return {
                'success': False,
                'error': str(e),
                'similarity_score': 0.0,
                'quality_label': 'ERROR'
            }
    
    def _determine_quality_label(self, similarity_score: float) -> str:
        """
        Determine quality label based on similarity score
        
        Args:
            similarity_score: Similarity score between 0 and 1
            
        Returns:
            Quality label: HIGH, MEDIUM, or LOW
        """
        if similarity_score >= 0.8:
            return 'HIGH'
        elif similarity_score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def compare_with_retry(self, sentence1: str, sentence2: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Compare sentences with retry logic
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            max_retries: Maximum number of retry attempts
            
        Returns:
            Comparison results
        """
        for attempt in range(max_retries):
            result = self.compare_sentences(sentence1, sentence2)
            
            if result['success']:
                return result
            
            if attempt < max_retries - 1:
                logger.warning(f"String comparison attempt {attempt + 1} failed, retrying...")
            else:
                logger.error(f"String comparison failed after {max_retries} attempts")
        
        return result
    
    def batch_compare(self, sentence_pairs: list) -> list:
        """
        Compare multiple sentence pairs
        
        Args:
            sentence_pairs: List of tuples (sentence1, sentence2)
            
        Returns:
            List of comparison results
        """
        results = []
        
        for i, (sentence1, sentence2) in enumerate(sentence_pairs):
            logger.info(f"Comparing pair {i + 1}/{len(sentence_pairs)}")
            result = self.compare_sentences(sentence1, sentence2)
            results.append(result)
        
        return results
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the string comparison service
        
        Returns:
            Service information or error
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                return {
                    'available': True,
                    'info': response.json()
                }
            else:
                return {
                    'available': False,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
