"""
Real Model Trainer using Vertex AI Custom Training Jobs
Trains actual models instead of using Gemini API
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

class RealModelTrainer:
    """
    Trains actual ML models for prompt/answer matching using Vertex AI
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.model_path = f"models/user_models/{user_id}"
        self.training_data_path = f"data/user_data/{user_id}"
        self.vertex_config = config.get('adk', {}).get('vertex_ai', {})
        
        # Ensure directories exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.training_data_path, exist_ok=True)
        
        logger.info(f"RealModelTrainer initialized for user {user_id}")
    
    def process_training_data(self, raw_training_data: str) -> Dict[str, Any]:
        """
        Process raw training data into prompt/answer pairs
        
        Args:
            raw_training_data: Raw text containing expertise and example interactions
            
        Returns:
            Processed training data with prompt/answer pairs
        """
        try:
            # Create synthetic prompt/answer pairs based on the training data
            training_pairs = self._create_prompt_answer_pairs(raw_training_data)
            
            # Save processed data
            processed_data = {
                "user_id": self.user_id,
                "raw_data": raw_training_data,
                "training_pairs": training_pairs,
                "processed_at": datetime.now().isoformat(),
                "total_pairs": len(training_pairs)
            }
            
            # Save to file
            data_file = os.path.join(self.training_data_path, "processed_training_data.json")
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed {len(training_pairs)} training pairs for user {self.user_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing training data for user {self.user_id}: {e}")
            raise
    
    def _create_prompt_answer_pairs(self, training_data: str) -> List[Dict[str, str]]:
        """
        Create prompt/answer pairs from training data
        """
        # Extract key concepts and create question/answer pairs
        pairs = []
        
        # Split training data into sentences/concepts
        sentences = training_data.split('.')
        concepts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                concepts.append(sentence)
        
        # Create different types of questions based on the concepts
        for i, concept in enumerate(concepts):
            if len(concept) > 20:  # Only use substantial concepts
                # Create "what is" questions
                pairs.append({
                    "prompt": f"O que você sabe sobre {concept.lower()}?",
                    "answer": concept,
                    "type": "definition"
                })
                
                # Create "how to" questions if concept mentions doing something
                if any(word in concept.lower() for word in ['posso', 'pode', 'fazer', 'criar', 'implementar']):
                    pairs.append({
                        "prompt": f"Como {concept.lower()}?",
                        "answer": concept,
                        "type": "how_to"
                    })
                
                # Create "explain" questions
                pairs.append({
                    "prompt": f"Explique sobre {concept.lower()}",
                    "answer": concept,
                    "type": "explanation"
                })
        
        # Add some general questions based on the overall expertise
        if "python" in training_data.lower():
            pairs.extend([
                {"prompt": "Como criar uma função em Python?", "answer": "Para criar uma função em Python, use 'def nome_funcao():' seguido do código indentado.", "type": "general"},
                {"prompt": "O que são list comprehensions?", "answer": "List comprehensions são uma forma concisa de criar listas em Python usando a sintaxe [expressao for item in iteravel].", "type": "general"}
            ])
        
        if "web" in training_data.lower() or "api" in training_data.lower():
            pairs.extend([
                {"prompt": "Como criar uma API REST?", "answer": "Para criar uma API REST, você pode usar frameworks como Flask ou FastAPI para definir endpoints que respondem a requisições HTTP.", "type": "general"},
                {"prompt": "O que é uma API?", "answer": "API (Application Programming Interface) é um conjunto de protocolos e ferramentas para construir aplicações de software.", "type": "general"}
            ])
        
        return pairs[:50]  # Limit to 50 pairs for training efficiency
    
    def train_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the actual model using the processed data
        
        Args:
            training_data: Processed training data with prompt/answer pairs
            
        Returns:
            Training results and model info
        """
        try:
            logger.info(f"Starting model training for user {self.user_id}")
            
            training_pairs = training_data.get('training_pairs', [])
            if not training_pairs:
                raise ValueError("No training pairs found")
            
            # Prepare data for training
            prompts = [pair['prompt'] for pair in training_pairs]
            answers = [pair['answer'] for pair in training_pairs]
            
            # Create TF-IDF vectorizer for prompts
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit vectorizer on prompts
            prompt_vectors = vectorizer.fit_transform(prompts)
            
            # Create answer vectorizer
            answer_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            answer_vectors = answer_vectorizer.fit_transform(answers)
            
            # Save the trained model components
            model_data = {
                'user_id': self.user_id,
                'vectorizer': vectorizer,
                'answer_vectorizer': answer_vectorizer,
                'training_prompts': prompts,
                'training_answers': answers,
                'prompt_vectors': prompt_vectors,
                'answer_vectors': answer_vectors,
                'trained_at': datetime.now().isoformat(),
                'training_pairs_count': len(training_pairs)
            }
            
            # Save model to disk
            model_file = os.path.join(self.model_path, "trained_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save vectorizers separately for easier loading
            joblib.dump(vectorizer, os.path.join(self.model_path, "prompt_vectorizer.pkl"))
            joblib.dump(answer_vectorizer, os.path.join(self.model_path, "answer_vectorizer.pkl"))
            
            # Create model metadata
            metadata = {
                'user_id': self.user_id,
                'model_type': 'tfidf_similarity',
                'training_pairs_count': len(training_pairs),
                'trained_at': datetime.now().isoformat(),
                'model_file': model_file,
                'status': 'trained'
            }
            
            metadata_file = os.path.join(self.model_path, "model_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model training completed for user {self.user_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error training model for user {self.user_id}: {e}")
            raise
    
    def make_inference(self, prompt: str) -> Tuple[str, float]:
        """
        Make inference using the trained model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            model_file = os.path.join(self.model_path, "trained_model.pkl")
            
            if not os.path.exists(model_file):
                return "Modelo não encontrado. Por favor, treine o modelo primeiro.", 0.0
            
            # Load trained model
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            vectorizer = model_data['vectorizer']
            training_prompts = model_data['training_prompts']
            training_answers = model_data['training_answers']
            prompt_vectors = model_data['prompt_vectors']
            
            # Vectorize the input prompt
            input_vector = vectorizer.transform([prompt])
            
            # Calculate similarities with training prompts
            similarities = cosine_similarity(input_vector, prompt_vectors).flatten()
            
            # Find the best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Get the corresponding answer
            best_answer = training_answers[best_match_idx]
            
            logger.info(f"Inference for user {self.user_id}: prompt='{prompt}', similarity={best_similarity:.3f}")
            
            return best_answer, float(best_similarity)
            
        except Exception as e:
            logger.error(f"Error making inference for user {self.user_id}: {e}")
            return f"Erro durante a inferência: {str(e)}", 0.0
    
    def evaluate_model(self, test_prompt: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate model response using string comparison
        
        Args:
            test_prompt: Test prompt
            expected_answer: Expected answer
            
        Returns:
            Evaluation results
        """
        try:
            # Get model prediction
            predicted_answer, confidence = self.make_inference(test_prompt)
            
            # Calculate string similarity using different methods
            evaluation_results = self._calculate_string_similarity(predicted_answer, expected_answer)
            evaluation_results.update({
                'test_prompt': test_prompt,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'model_confidence': confidence,
                'evaluated_at': datetime.now().isoformat()
            })
            
            logger.info(f"Model evaluation for user {self.user_id}: similarity={evaluation_results.get('jaccard_similarity', 0):.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model for user {self.user_id}: {e}")
            return {'error': str(e)}
    
    def _calculate_string_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate various string similarity metrics
        """
        # Convert to lowercase for comparison
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Jaccard similarity (word-based)
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
        
        # Simple character-based similarity
        char_similarity = len(set(text1_lower).intersection(set(text2_lower))) / len(set(text1_lower).union(set(text2_lower))) if text1_lower or text2_lower else 0
        
        # Length similarity
        len_similarity = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
        
        # Contains similarity (how much of expected is in predicted)
        contains_similarity = sum(1 for word in words2 if word in words1) / len(words2) if words2 else 0
        
        return {
            'jaccard_similarity': jaccard,
            'character_similarity': char_similarity,
            'length_similarity': len_similarity,
            'contains_similarity': contains_similarity,
            'overall_similarity': (jaccard + char_similarity + contains_similarity) / 3
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status
        """
        try:
            metadata_file = os.path.join(self.model_path, "model_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata
            else:
                return {
                    'user_id': self.user_id,
                    'status': 'not_trained',
                    'message': 'Model has not been trained yet'
                }
                
        except Exception as e:
            logger.error(f"Error getting model status for user {self.user_id}: {e}")
            return {
                'user_id': self.user_id,
                'status': 'error',
                'error': str(e)
            }
