"""Embedding and similarity services."""

import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
import structlog

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = structlog.get_logger()

class SimilarityService:
    """Service for embeddings, topic detection, and similarity matching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._topic_centroids: Dict[str, np.ndarray] = {}
        
        # Default topic keywords for fallback detection
        self.topic_keywords = {
            "coding": ["code", "programming", "function", "bug", "debug", "algorithm", "python", "javascript", "sql"],
            "math": ["calculate", "equation", "solve", "formula", "mathematics", "algebra", "geometry"],
            "science": ["research", "experiment", "hypothesis", "theory", "scientific", "analysis"],
            "writing": ["essay", "article", "content", "draft", "editing", "grammar", "style"],
            "business": ["strategy", "marketing", "sales", "revenue", "profit", "management"]
        }
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence_transformers_not_available", fallback="keyword_only")
            return None
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            # Fallback: create dummy embedding based on text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # Convert hash to pseudo-embedding (384 dimensions like all-MiniLM-L6-v2)
            embedding = []
            for i in range(0, min(len(text_hash), 96), 1):  # Use hash chars for variation
                val = int(text_hash[i:i+1], 16) / 15.0 - 0.5  # Normalize to [-0.5, 0.5]
                embedding.extend([val] * 4)  # Repeat to get 384 dims
            # Pad to 384 if needed
            while len(embedding) < 384:
                embedding.append(0.0)
            return embedding[:384]
        
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def detect_topic(self, prompt: str, topic_hint: Optional[str] = None) -> str:
        """Detect topic from prompt using multiple strategies."""
        
        # Strategy 1: Use provided hint if available
        if topic_hint:
            normalized_hint = topic_hint.strip().lower()
            if normalized_hint in self.topic_keywords:
                return normalized_hint
            # Map common variations
            hint_mapping = {
                "programming": "coding",
                "development": "coding",
                "mathematics": "math",
                "calculation": "math",
                "literature": "writing",
                "composition": "writing"
            }
            return hint_mapping.get(normalized_hint, "general")
        
        # Strategy 2: Embedding-based detection using centroids
        if self._topic_centroids and SENTENCE_TRANSFORMERS_AVAILABLE:
            prompt_embedding = self.generate_embedding(prompt)
            best_topic, best_similarity = self._find_best_topic_by_embedding(prompt_embedding)
            
            # Use embedding result if confidence is high enough
            if best_similarity >= 0.6:
                logger.info("topic_detected_by_embedding", topic=best_topic, similarity=best_similarity)
                return best_topic
        
        # Strategy 3: Keyword-based fallback
        detected_topic = self._detect_topic_by_keywords(prompt)
        logger.info("topic_detected_by_keywords", topic=detected_topic)
        return detected_topic
    
    def _detect_topic_by_keywords(self, prompt: str) -> str:
        """Detect topic using keyword matching."""
        prompt_lower = prompt.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return "general"
    
    def _find_best_topic_by_embedding(self, prompt_embedding: List[float]) -> Tuple[str, float]:
        """Find best topic by comparing to stored centroids."""
        best_topic = "general"
        best_similarity = 0.0
        
        for topic, centroid in self._topic_centroids.items():
            similarity = self.cosine_similarity(prompt_embedding, centroid.tolist())
            if similarity > best_similarity:
                best_similarity = similarity
                best_topic = topic
        
        return best_topic, best_similarity
    
    def update_topic_centroids(self, topic_embeddings: Dict[str, List[List[float]]]):
        """Update topic centroids from collections of embeddings."""
        for topic, embeddings in topic_embeddings.items():
            if embeddings:
                embeddings_array = np.array(embeddings)
                centroid = np.mean(embeddings_array, axis=0)
                self._topic_centroids[topic] = centroid
                logger.info("topic_centroid_updated", topic=topic, num_embeddings=len(embeddings))
    
    def find_similar_prompts(self, 
                           target_embedding: List[float], 
                           candidate_embeddings: List[List[float]], 
                           threshold: float = 0.90) -> List[int]:
        """Find indices of similar prompts above threshold."""
        similar_indices = []
        
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(target_embedding, candidate_emb)
            if similarity >= threshold:
                similar_indices.append(i)
        
        return similar_indices
    
    def compute_prompt_hash(self, prompt: str) -> str:
        """Compute SHA-256 hash of normalized prompt."""
        normalized_prompt = prompt.strip().lower()
        return hashlib.sha256(normalized_prompt.encode('utf-8')).hexdigest()
    
    def embedding_to_json(self, embedding: List[float]) -> str:
        """Convert embedding to JSON string for storage."""
        return json.dumps(embedding)
    
    def embedding_from_json(self, embedding_str: str) -> List[float]:
        """Convert embedding from JSON string."""
        return json.loads(embedding_str)
    
    def get_available_topics(self) -> List[str]:
        """Get list of topics with available centroids."""
        return list(self._topic_centroids.keys())
    
    def add_topic_examples(self, topic: str, prompts: List[str]):
        """Add example prompts for a topic to update its centroid."""
        embeddings = [self.generate_embedding(prompt) for prompt in prompts]
        
        if topic in self._topic_centroids:
            # Update existing centroid
            existing_centroid = self._topic_centroids[topic]
            all_embeddings = embeddings + [existing_centroid.tolist()]
            new_centroid = np.mean(np.array(all_embeddings), axis=0)
            self._topic_centroids[topic] = new_centroid
        else:
            # Create new centroid
            centroid = np.mean(np.array(embeddings), axis=0)
            self._topic_centroids[topic] = centroid
        
        logger.info("topic_examples_added", topic=topic, num_examples=len(prompts))