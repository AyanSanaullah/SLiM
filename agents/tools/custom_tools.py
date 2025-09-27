"""
Custom tools for Google ADK agents
Provides specialized tools for model fine-tuning and management
"""

from google.adk.tools import Tool
from typing import Dict, List, Any, Optional
import json
import os
import logging
from datetime import datetime
import hashlib
import re

logger = logging.getLogger(__name__)

class CustomTools:
    """
    Collection of custom tools for ADK agents
    """
    
    @staticmethod
    def create_data_validation_tool() -> Tool:
        """Creates a tool for validating training data quality"""
        
        def validate_data_quality(raw_data: str, user_id: str) -> str:
            """
            Validates the quality and format of training data
            
            Args:
                raw_data: Raw training data string
                user_id: User identifier
                
            Returns:
                JSON string with validation results
            """
            try:
                logger.info(f"Validating data quality for user {user_id}")
                
                validation_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "validation_status": "success",
                    "data_metrics": {
                        "total_characters": len(raw_data),
                        "total_words": len(raw_data.split()),
                        "total_sentences": len(re.split(r'[.!?]+', raw_data)),
                        "total_paragraphs": len(raw_data.split('\n\n')),
                        "average_sentence_length": len(raw_data.split()) / max(1, len(re.split(r'[.!?]+', raw_data))),
                        "data_diversity_score": CustomTools._calculate_diversity_score(raw_data)
                    },
                    "quality_checks": {
                        "minimum_length": len(raw_data) >= 100,
                        "has_variety": len(set(raw_data.lower().split())) > 20,
                        "proper_formatting": not bool(re.search(r'[^\w\s.,!?;:\'"-]', raw_data)),
                        "no_excessive_whitespace": not bool(re.search(r'\s{3,}', raw_data))
                    },
                    "recommendations": []
                }
                
                # Add recommendations based on validation
                if validation_result["data_metrics"]["total_words"] < 50:
                    validation_result["recommendations"].append("Consider adding more training data for better model performance")
                
                if validation_result["data_metrics"]["data_diversity_score"] < 0.5:
                    validation_result["recommendations"].append("Data lacks diversity - consider adding varied examples")
                
                if not validation_result["quality_checks"]["proper_formatting"]:
                    validation_result["recommendations"].append("Clean up special characters in the data")
                
                validation_result["overall_quality_score"] = CustomTools._calculate_overall_quality_score(validation_result)
                
                return json.dumps(validation_result)
                
            except Exception as e:
                logger.error(f"Error validating data quality for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "validation_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="validate_data_quality",
            description="Validates the quality and format of training data",
            function=validate_data_quality
        )
    
    @staticmethod
    def create_model_optimization_tool() -> Tool:
        """Creates a tool for optimizing model parameters"""
        
        def optimize_model_parameters(data_info: str, user_id: str) -> str:
            """
            Optimizes model parameters based on data characteristics
            
            Args:
                data_info: JSON string with data information
                user_id: User identifier
                
            Returns:
                JSON string with optimized parameters
            """
            try:
                logger.info(f"Optimizing model parameters for user {user_id}")
                
                data_obj = json.loads(data_info)
                data_metrics = data_obj.get('data_metrics', {})
                
                # Calculate optimal parameters based on data characteristics
                total_words = data_metrics.get('total_words', 0)
                avg_sentence_length = data_metrics.get('average_sentence_length', 0)
                diversity_score = data_metrics.get('data_diversity_score', 0)
                
                optimized_params = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "optimization_status": "success",
                    "optimized_parameters": {
                        "learning_rate": CustomTools._optimize_learning_rate(total_words, diversity_score),
                        "batch_size": CustomTools._optimize_batch_size(avg_sentence_length),
                        "num_epochs": CustomTools._optimize_epochs(total_words),
                        "warmup_steps": CustomTools._optimize_warmup_steps(total_words),
                        "max_length": CustomTools._optimize_max_length(avg_sentence_length),
                        "dropout_rate": CustomTools._optimize_dropout_rate(diversity_score),
                        "weight_decay": CustomTools._optimize_weight_decay(total_words)
                    },
                    "optimization_reasoning": {
                        "data_size_factor": "Large dataset" if total_words > 1000 else "Small dataset",
                        "complexity_factor": "High complexity" if diversity_score > 0.7 else "Low complexity",
                        "recommended_approach": CustomTools._recommend_training_approach(total_words, diversity_score)
                    }
                }
                
                return json.dumps(optimized_params)
                
            except Exception as e:
                logger.error(f"Error optimizing parameters for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "optimization_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="optimize_model_parameters",
            description="Optimizes model parameters based on data characteristics",
            function=optimize_model_parameters
        )
    
    @staticmethod
    def create_performance_monitor_tool() -> Tool:
        """Creates a tool for monitoring model performance"""
        
        def monitor_model_performance(model_id: str, user_id: str) -> str:
            """
            Monitors model performance and generates reports
            
            Args:
                model_id: Model identifier
                user_id: User identifier
                
            Returns:
                JSON string with performance metrics
            """
            try:
                logger.info(f"Monitoring performance for model {model_id} of user {user_id}")
                
                # Simulate performance monitoring
                performance_metrics = {
                    "user_id": user_id,
                    "model_id": model_id,
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_status": "active",
                    "performance_metrics": {
                        "inference_speed": CustomTools._simulate_inference_speed(),
                        "memory_usage": CustomTools._simulate_memory_usage(),
                        "accuracy_score": CustomTools._simulate_accuracy_score(),
                        "throughput": CustomTools._simulate_throughput(),
                        "error_rate": CustomTools._simulate_error_rate()
                    },
                    "health_indicators": {
                        "cpu_utilization": CustomTools._simulate_cpu_usage(),
                        "memory_utilization": CustomTools._simulate_memory_usage(),
                        "disk_usage": CustomTools._simulate_disk_usage(),
                        "network_latency": CustomTools._simulate_network_latency()
                    },
                    "quality_metrics": {
                        "response_relevance": CustomTools._simulate_relevance_score(),
                        "response_coherence": CustomTools._simulate_coherence_score(),
                        "response_consistency": CustomTools._simulate_consistency_score(),
                        "user_satisfaction": CustomTools._simulate_satisfaction_score()
                    },
                    "alerts": CustomTools._generate_performance_alerts(performance_metrics),
                    "recommendations": CustomTools._generate_performance_recommendations(performance_metrics)
                }
                
                return json.dumps(performance_metrics)
                
            except Exception as e:
                logger.error(f"Error monitoring performance for model {model_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "model_id": model_id,
                    "monitoring_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="monitor_model_performance",
            description="Monitors model performance and generates reports",
            function=monitor_model_performance
        )
    
    @staticmethod
    def create_data_augmentation_tool() -> Tool:
        """Creates a tool for data augmentation"""
        
        def augment_training_data(original_data: str, user_id: str) -> str:
            """
            Augments training data to improve model performance
            
            Args:
                original_data: Original training data
                user_id: User identifier
                
            Returns:
                JSON string with augmented data
            """
            try:
                logger.info(f"Augmenting training data for user {user_id}")
                
                # Parse original data
                sentences = re.split(r'[.!?]+', original_data)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                augmented_data = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "augmentation_status": "success",
                    "original_sentences": len(sentences),
                    "augmentation_techniques": [],
                    "augmented_sentences": []
                }
                
                # Apply augmentation techniques
                for sentence in sentences:
                    augmented_data["augmented_sentences"].append(sentence)  # Original
                    
                    # Synonym replacement
                    if len(sentence.split()) > 3:
                        augmented_sentence = CustomTools._apply_synonym_replacement(sentence)
                        if augmented_sentence != sentence:
                            augmented_data["augmented_sentences"].append(augmented_sentence)
                            if "synonym_replacement" not in augmented_data["augmentation_techniques"]:
                                augmented_data["augmentation_techniques"].append("synonym_replacement")
                    
                    # Paraphrasing
                    if len(sentence.split()) > 5:
                        paraphrased = CustomTools._apply_paraphrasing(sentence)
                        if paraphrased != sentence:
                            augmented_data["augmented_sentences"].append(paraphrased)
                            if "paraphrasing" not in augmented_data["augmentation_techniques"]:
                                augmented_data["augmentation_techniques"].append("paraphrasing")
                
                augmented_data["augmented_sentences_count"] = len(augmented_data["augmented_sentences"])
                augmented_data["augmentation_ratio"] = augmented_data["augmented_sentences_count"] / augmented_data["original_sentences"]
                
                return json.dumps(augmented_data)
                
            except Exception as e:
                logger.error(f"Error augmenting data for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "augmentation_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="augment_training_data",
            description="Augments training data to improve model performance",
            function=augment_training_data
        )
    
    @staticmethod
    def create_model_comparison_tool() -> Tool:
        """Creates a tool for comparing different models"""
        
        def compare_models(model_configs: str, user_id: str) -> str:
            """
            Compares different model configurations
            
            Args:
                model_configs: JSON string with model configurations
                user_id: User identifier
                
            Returns:
                JSON string with comparison results
            """
            try:
                logger.info(f"Comparing models for user {user_id}")
                
                configs_obj = json.loads(model_configs)
                models = configs_obj.get('models', [])
                
                comparison_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "comparison_status": "success",
                    "models_compared": len(models),
                    "comparison_metrics": [],
                    "recommendations": []
                }
                
                for i, model in enumerate(models):
                    model_metrics = {
                        "model_id": model.get('model_id', f'model_{i}'),
                        "base_model": model.get('base_model', 'unknown'),
                        "parameters": model.get('parameters', {}),
                        "performance_scores": {
                            "accuracy": CustomTools._simulate_model_accuracy(model),
                            "speed": CustomTools._simulate_model_speed(model),
                            "memory_efficiency": CustomTools._simulate_memory_efficiency(model),
                            "overall_score": 0.0
                        }
                    }
                    
                    # Calculate overall score
                    scores = model_metrics["performance_scores"]
                    model_metrics["performance_scores"]["overall_score"] = (
                        scores["accuracy"] * 0.4 + 
                        scores["speed"] * 0.3 + 
                        scores["memory_efficiency"] * 0.3
                    )
                    
                    comparison_result["comparison_metrics"].append(model_metrics)
                
                # Sort by overall score
                comparison_result["comparison_metrics"].sort(
                    key=lambda x: x["performance_scores"]["overall_score"], 
                    reverse=True
                )
                
                # Generate recommendations
                best_model = comparison_result["comparison_metrics"][0]
                comparison_result["recommendations"] = [
                    f"Recommended model: {best_model['model_id']} (score: {best_model['performance_scores']['overall_score']:.2f})",
                    f"Best for accuracy: {max(comparison_result['comparison_metrics'], key=lambda x: x['performance_scores']['accuracy'])['model_id']}",
                    f"Best for speed: {max(comparison_result['comparison_metrics'], key=lambda x: x['performance_scores']['speed'])['model_id']}"
                ]
                
                return json.dumps(comparison_result)
                
            except Exception as e:
                logger.error(f"Error comparing models for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "comparison_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="compare_models",
            description="Compares different model configurations",
            function=compare_models
        )
    
    # Helper methods for calculations and simulations
    @staticmethod
    def _calculate_diversity_score(text: str) -> float:
        """Calculate diversity score based on unique words and sentence variety"""
        words = text.lower().split()
        unique_words = set(words)
        sentences = re.split(r'[.!?]+', text)
        
        word_diversity = len(unique_words) / len(words) if words else 0
        sentence_diversity = len(set(len(s.split()) for s in sentences)) / len(sentences) if sentences else 0
        
        return (word_diversity + sentence_diversity) / 2
    
    @staticmethod
    def _calculate_overall_quality_score(validation_result: Dict) -> float:
        """Calculate overall quality score from validation results"""
        quality_checks = validation_result.get('quality_checks', {})
        checks_passed = sum(quality_checks.values())
        total_checks = len(quality_checks)
        
        data_metrics = validation_result.get('data_metrics', {})
        diversity_score = data_metrics.get('data_diversity_score', 0)
        
        return (checks_passed / total_checks * 0.7 + diversity_score * 0.3) if total_checks > 0 else 0
    
    @staticmethod
    def _optimize_learning_rate(total_words: int, diversity_score: float) -> float:
        """Optimize learning rate based on data characteristics"""
        if total_words < 100:
            return 1e-4  # Lower LR for small datasets
        elif diversity_score > 0.7:
            return 5e-5  # Lower LR for complex data
        else:
            return 2e-4  # Higher LR for simple data
    
    @staticmethod
    def _optimize_batch_size(avg_sentence_length: float) -> int:
        """Optimize batch size based on sentence length"""
        if avg_sentence_length < 10:
            return 32
        elif avg_sentence_length < 20:
            return 16
        else:
            return 8
    
    @staticmethod
    def _optimize_epochs(total_words: int) -> int:
        """Optimize number of epochs based on data size"""
        if total_words < 100:
            return 10
        elif total_words < 500:
            return 5
        else:
            return 3
    
    @staticmethod
    def _optimize_warmup_steps(total_words: int) -> int:
        """Optimize warmup steps based on data size"""
        return max(50, min(500, total_words // 10))
    
    @staticmethod
    def _optimize_max_length(avg_sentence_length: float) -> int:
        """Optimize max length based on sentence length"""
        return min(512, max(128, int(avg_sentence_length * 4)))
    
    @staticmethod
    def _optimize_dropout_rate(diversity_score: float) -> float:
        """Optimize dropout rate based on diversity"""
        return max(0.1, min(0.5, 0.3 + (1 - diversity_score) * 0.2))
    
    @staticmethod
    def _optimize_weight_decay(total_words: int) -> float:
        """Optimize weight decay based on data size"""
        if total_words < 100:
            return 0.01
        elif total_words < 500:
            return 0.005
        else:
            return 0.001
    
    @staticmethod
    def _recommend_training_approach(total_words: int, diversity_score: float) -> str:
        """Recommend training approach based on data characteristics"""
        if total_words < 100:
            return "Use transfer learning with frozen layers"
        elif diversity_score < 0.3:
            return "Use fine-tuning with careful regularization"
        else:
            return "Use full fine-tuning with data augmentation"
    
    # Simulation methods for monitoring
    @staticmethod
    def _simulate_inference_speed() -> float:
        import random
        return random.uniform(0.1, 2.0)
    
    @staticmethod
    def _simulate_memory_usage() -> float:
        import random
        return random.uniform(100, 2000)  # MB
    
    @staticmethod
    def _simulate_accuracy_score() -> float:
        import random
        return random.uniform(0.7, 0.95)
    
    @staticmethod
    def _simulate_throughput() -> float:
        import random
        return random.uniform(10, 100)  # requests/second
    
    @staticmethod
    def _simulate_error_rate() -> float:
        import random
        return random.uniform(0.001, 0.05)
    
    @staticmethod
    def _simulate_cpu_usage() -> float:
        import random
        return random.uniform(20, 90)  # percentage
    
    @staticmethod
    def _simulate_disk_usage() -> float:
        import random
        return random.uniform(1, 10)  # GB
    
    @staticmethod
    def _simulate_network_latency() -> float:
        import random
        return random.uniform(10, 100)  # ms
    
    @staticmethod
    def _simulate_relevance_score() -> float:
        import random
        return random.uniform(0.8, 0.95)
    
    @staticmethod
    def _simulate_coherence_score() -> float:
        import random
        return random.uniform(0.75, 0.92)
    
    @staticmethod
    def _simulate_consistency_score() -> float:
        import random
        return random.uniform(0.7, 0.9)
    
    @staticmethod
    def _simulate_satisfaction_score() -> float:
        import random
        return random.uniform(0.8, 0.98)
    
    @staticmethod
    def _generate_performance_alerts(metrics: Dict) -> List[str]:
        """Generate performance alerts based on metrics"""
        alerts = []
        perf_metrics = metrics.get('performance_metrics', {})
        
        if perf_metrics.get('error_rate', 0) > 0.02:
            alerts.append("High error rate detected")
        if perf_metrics.get('inference_speed', 0) > 1.5:
            alerts.append("Slow inference speed")
        if metrics.get('health_indicators', {}).get('memory_utilization', 0) > 85:
            alerts.append("High memory utilization")
        
        return alerts
    
    @staticmethod
    def _generate_performance_recommendations(metrics: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        perf_metrics = metrics.get('performance_metrics', {})
        
        if perf_metrics.get('accuracy_score', 0) < 0.8:
            recommendations.append("Consider retraining with more data")
        if perf_metrics.get('throughput', 0) < 20:
            recommendations.append("Consider scaling up resources")
        if metrics.get('quality_metrics', {}).get('response_consistency', 0) < 0.8:
            recommendations.append("Improve model consistency through regularization")
        
        return recommendations
    
    # Data augmentation methods
    @staticmethod
    def _apply_synonym_replacement(sentence: str) -> str:
        """Apply synonym replacement to a sentence"""
        # Simple synonym replacement simulation
        synonyms = {
            'good': 'great',
            'bad': 'terrible',
            'big': 'large',
            'small': 'tiny',
            'happy': 'joyful',
            'sad': 'unhappy'
        }
        
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                words[i] = synonyms[word.lower()]
                break  # Replace only one word
        
        return ' '.join(words)
    
    @staticmethod
    def _apply_paraphrasing(sentence: str) -> str:
        """Apply paraphrasing to a sentence"""
        # Simple paraphrasing simulation
        paraphrases = {
            'I am': "I'm",
            'you are': "you're",
            'it is': "it's",
            'we are': "we're",
            'they are': "they're"
        }
        
        paraphrased = sentence
        for original, paraphrase in paraphrases.items():
            if original in paraphrased.lower():
                paraphrased = paraphrased.replace(original, paraphrase)
                break
        
        return paraphrased
    
    @staticmethod
    def _simulate_model_accuracy(model: Dict) -> float:
        """Simulate model accuracy based on configuration"""
        import random
        base_model = model.get('base_model', '')
        
        # Adjust accuracy based on base model
        if 'distilbert' in base_model.lower():
            return random.uniform(0.75, 0.85)
        elif 'tinybert' in base_model.lower():
            return random.uniform(0.70, 0.80)
        else:
            return random.uniform(0.80, 0.90)
    
    @staticmethod
    def _simulate_model_speed(model: Dict) -> float:
        """Simulate model speed based on configuration"""
        import random
        base_model = model.get('base_model', '')
        
        # Smaller models are faster
        if 'tiny' in base_model.lower():
            return random.uniform(0.8, 0.95)
        elif 'mobile' in base_model.lower():
            return random.uniform(0.7, 0.9)
        else:
            return random.uniform(0.6, 0.8)
    
    @staticmethod
    def _simulate_memory_efficiency(model: Dict) -> float:
        """Simulate memory efficiency based on configuration"""
        import random
        base_model = model.get('base_model', '')
        
        # Smaller models use less memory
        if 'tiny' in base_model.lower():
            return random.uniform(0.85, 0.95)
        elif 'mobile' in base_model.lower():
            return random.uniform(0.75, 0.85)
        else:
            return random.uniform(0.65, 0.75)
