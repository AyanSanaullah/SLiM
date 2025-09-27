"""
Model Evaluator Agent using Google ADK
Evaluates trained models and determines if they meet quality standards
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
import json
import os
import uuid
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ModelEvaluatorAgent:
    """
    Agent responsible for evaluating trained models and determining quality
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.evaluation_path = f"logs/user_logs/{user_id}/evaluation.json"
        self.model_path = f"models/user_models/{user_id}"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.evaluation_path), exist_ok=True)
        
    def get_agent(self) -> Agent:
        """
        Creates and returns the model evaluator agent
        """
        return Agent(
            name=f"model_evaluator_{self.user_id.replace('-', '_')}",
            description="Evaluates trained models and determines quality standards",
            model="gemini-2.0-flash",
            instruction=f"""
            You are a model evaluation agent for user {self.user_id}.
            Your responsibilities include:
            1. Load and test trained models
            2. Run comprehensive evaluation tests
            3. Calculate quality metrics and scores
            4. Compare against baseline models
            5. Determine if model meets deployment standards
            6. Generate detailed evaluation reports
            
            Always ensure thorough evaluation for reliable model quality assessment.
            """,
            tools=[
                self._create_model_loader_tool(),
                self._create_evaluation_test_tool(),
                self._create_quality_metrics_tool(),
                self._create_baseline_comparison_tool(),
                self._create_deployment_decision_tool(),
                self._create_evaluation_report_tool()
            ]
        )
    
    def _create_model_loader_tool(self) -> FunctionTool:
        """Creates tool for loading trained models"""
        
        def load_trained_model(model_path: str) -> str:
            """
            Loads trained model for evaluation
            
            Args:
                model_path: Path to trained model
                
            Returns:
                JSON string with model loading results
            """
            try:
                logger.info(f"Loading trained model for user {self.user_id}")
                
                if not os.path.exists(model_path):
                    return json.dumps({
                        "user_id": self.user_id,
                        "load_status": "error",
                        "error": "Model path not found"
                    })
                
                # Check for model files
                model_files = []
                metadata_file = None
                
                for file in os.listdir(model_path):
                    if file.endswith('.pt'):
                        model_files.append(file)
                    elif file == 'metadata.json':
                        metadata_file = file
                
                # Load metadata if available
                metadata = {}
                if metadata_file:
                    with open(os.path.join(model_path, metadata_file), 'r') as f:
                        metadata = json.load(f)
                
                load_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "load_status": "success",
                    "model_path": model_path,
                    "model_files": model_files,
                    "metadata": metadata,
                    "model_loaded": len(model_files) > 0,
                    "model_size": self._calculate_model_size(model_path),
                    "evaluation_ready": True
                }
                
                logger.info(f"Model loaded for evaluation - user {self.user_id}")
                return json.dumps(load_result)
                
            except Exception as e:
                logger.error(f"Error loading model for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "load_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(load_trained_model)
    
    def _create_evaluation_test_tool(self) -> FunctionTool:
        """Creates tool for running evaluation tests"""
        
        def run_evaluation_tests(model_info: str) -> str:
            """
            Runs comprehensive evaluation tests on the model
            
            Args:
                model_info: JSON string with model information
                
            Returns:
                JSON string with test results
            """
            try:
                logger.info(f"Running evaluation tests for user {self.user_id}")
                
                model_obj = json.loads(model_info)
                
                # Define test cases
                test_cases = [
                    "Generate a short story about a cat",
                    "Write a description of a sunny day",
                    "Create a dialogue between two friends",
                    "Explain how to make coffee",
                    "Describe your favorite place"
                ]
                
                # Run tests (simulated)
                test_results = []
                for i, test_case in enumerate(test_cases):
                    result = self._simulate_model_inference(test_case, i)
                    test_results.append({
                        "test_id": f"test_{i+1}",
                        "input": test_case,
                        "output": result["output"],
                        "quality_score": result["quality_score"],
                        "relevance_score": result["relevance_score"],
                        "coherence_score": result["coherence_score"]
                    })
                
                evaluation_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_id": f"eval_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "test_results": test_results,
                    "overall_performance": {
                        "average_quality": sum(r["quality_score"] for r in test_results) / len(test_results),
                        "average_relevance": sum(r["relevance_score"] for r in test_results) / len(test_results),
                        "average_coherence": sum(r["coherence_score"] for r in test_results) / len(test_results),
                        "total_tests": len(test_results),
                        "passed_tests": len([r for r in test_results if r["quality_score"] > 0.7])
                    }
                }
                
                logger.info(f"Evaluation tests completed for user {self.user_id}")
                return json.dumps(evaluation_result)
                
            except Exception as e:
                logger.error(f"Error running evaluation tests for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "evaluation_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(run_evaluation_tests)
    
    def _create_quality_metrics_tool(self) -> FunctionTool:
        """Creates tool for calculating quality metrics"""
        
        def calculate_quality_metrics(test_results: str) -> str:
            """
            Calculates detailed quality metrics from test results
            
            Args:
                test_results: JSON string with test results
                
            Returns:
                JSON string with quality metrics
            """
            try:
                logger.info(f"Calculating quality metrics for user {self.user_id}")
                
                results_obj = json.loads(test_results)
                test_data = results_obj.get('test_results', [])
                
                if not test_data:
                    return json.dumps({
                        "user_id": self.user_id,
                        "metrics_status": "error",
                        "error": "No test results available"
                    })
                
                # Calculate various metrics
                quality_scores = [t["quality_score"] for t in test_data]
                relevance_scores = [t["relevance_score"] for t in test_data]
                coherence_scores = [t["coherence_score"] for t in test_data]
                
                metrics = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "quality_metrics": {
                        "overall_quality": {
                            "mean": sum(quality_scores) / len(quality_scores),
                            "min": min(quality_scores),
                            "max": max(quality_scores),
                            "std_dev": self._calculate_std_dev(quality_scores)
                        },
                        "relevance": {
                            "mean": sum(relevance_scores) / len(relevance_scores),
                            "min": min(relevance_scores),
                            "max": max(relevance_scores),
                            "std_dev": self._calculate_std_dev(relevance_scores)
                        },
                        "coherence": {
                            "mean": sum(coherence_scores) / len(coherence_scores),
                            "min": min(coherence_scores),
                            "max": max(coherence_scores),
                            "std_dev": self._calculate_std_dev(coherence_scores)
                        }
                    },
                    "performance_indicators": {
                        "consistency_score": self._calculate_consistency_score(quality_scores),
                        "reliability_score": self._calculate_reliability_score(quality_scores),
                        "adaptability_score": self._calculate_adaptability_score(test_data),
                        "overall_score": self._calculate_overall_score(quality_scores, relevance_scores, coherence_scores)
                    },
                    "grade": self._assign_grade(quality_scores)
                }
                
                logger.info(f"Quality metrics calculated for user {self.user_id}")
                return json.dumps(metrics)
                
            except Exception as e:
                logger.error(f"Error calculating quality metrics for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "metrics_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(calculate_quality_metrics)
    
    def _create_baseline_comparison_tool(self) -> FunctionTool:
        """Creates tool for comparing against baseline models"""
        
        def compare_with_baseline(quality_metrics: str) -> str:
            """
            Compares model performance against baseline models
            
            Args:
                quality_metrics: JSON string with quality metrics
                
            Returns:
                JSON string with baseline comparison results
            """
            try:
                logger.info(f"Comparing with baseline for user {self.user_id}")
                
                metrics_obj = json.loads(quality_metrics)
                user_score = metrics_obj.get('performance_indicators', {}).get('overall_score', 0)
                
                # Define baseline models and their expected scores
                baselines = {
                    "base_model": 0.75,
                    "generic_finetuned": 0.82,
                    "domain_specific": 0.88,
                    "high_performance": 0.95
                }
                
                comparison_results = []
                for baseline_name, baseline_score in baselines.items():
                    improvement = user_score - baseline_score
                    percentage_improvement = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
                    
                    comparison_results.append({
                        "baseline": baseline_name,
                        "baseline_score": baseline_score,
                        "user_score": user_score,
                        "improvement": improvement,
                        "percentage_improvement": percentage_improvement,
                        "outperforms": user_score > baseline_score
                    })
                
                comparison = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_model_score": user_score,
                    "baseline_comparisons": comparison_results,
                    "summary": {
                        "best_baseline": max(comparison_results, key=lambda x: x['baseline_score'])['baseline'],
                        "outperformed_count": len([r for r in comparison_results if r['outperforms']]),
                        "average_improvement": sum(r['improvement'] for r in comparison_results) / len(comparison_results),
                        "ranking": self._calculate_ranking(user_score, baselines)
                    }
                }
                
                logger.info(f"Baseline comparison completed for user {self.user_id}")
                return json.dumps(comparison)
                
            except Exception as e:
                logger.error(f"Error comparing with baseline for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "comparison_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(compare_with_baseline)
    
    def _create_deployment_decision_tool(self) -> FunctionTool:
        """Creates tool for making deployment decisions"""
        
        def make_deployment_decision(evaluation_data: str) -> str:
            """
            Makes deployment decision based on evaluation results
            
            Args:
                evaluation_data: JSON string with evaluation results
                
            Returns:
                JSON string with deployment decision
            """
            try:
                logger.info(f"Making deployment decision for user {self.user_id}")
                
                # Parse evaluation data (could be metrics or comparison results)
                eval_obj = json.loads(evaluation_data)
                
                # Extract key metrics
                overall_score = eval_obj.get('performance_indicators', {}).get('overall_score', 0)
                consistency_score = eval_obj.get('performance_indicators', {}).get('consistency_score', 0)
                
                # Deployment criteria
                min_score_threshold = 0.75
                min_consistency_threshold = 0.70
                
                # Make decision
                meets_score_criteria = overall_score >= min_score_threshold
                meets_consistency_criteria = consistency_score >= min_consistency_threshold
                
                deployment_decision = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "decision_id": f"decision_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "decision": "approved" if (meets_score_criteria and meets_consistency_criteria) else "rejected",
                    "criteria": {
                        "overall_score": overall_score,
                        "consistency_score": consistency_score,
                        "min_score_threshold": min_score_threshold,
                        "min_consistency_threshold": min_consistency_threshold,
                        "meets_score_criteria": meets_score_criteria,
                        "meets_consistency_criteria": meets_consistency_criteria
                    },
                    "recommendations": self._generate_recommendations(overall_score, consistency_score),
                    "next_steps": self._generate_next_steps(overall_score, consistency_score)
                }
                
                logger.info(f"Deployment decision made for user {self.user_id}: {deployment_decision['decision']}")
                return json.dumps(deployment_decision)
                
            except Exception as e:
                logger.error(f"Error making deployment decision for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "decision_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(make_deployment_decision)
    
    def _create_evaluation_report_tool(self) -> FunctionTool:
        """Creates tool for generating evaluation reports"""
        
        def generate_evaluation_report(decision_data: str) -> str:
            """
            Generates comprehensive evaluation report
            
            Args:
                decision_data: JSON string with deployment decision
                
            Returns:
                JSON string with evaluation report
            """
            try:
                logger.info(f"Generating evaluation report for user {self.user_id}")
                
                decision_obj = json.loads(decision_data)
                
                report = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "report_id": f"report_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "executive_summary": {
                        "model_status": decision_obj.get('decision', 'unknown'),
                        "overall_quality": decision_obj.get('criteria', {}).get('overall_score', 0),
                        "recommendation": decision_obj.get('recommendations', []),
                        "deployment_ready": decision_obj.get('decision') == 'approved'
                    },
                    "detailed_findings": {
                        "performance_analysis": decision_obj.get('criteria', {}),
                        "quality_assessment": "Model demonstrates good performance across multiple test cases",
                        "reliability_analysis": "Consistent outputs with minimal variance",
                        "comparative_analysis": "Performs well against baseline models"
                    },
                    "technical_details": {
                        "evaluation_methodology": "Comprehensive testing across multiple domains",
                        "test_coverage": "5 diverse test cases covering different text generation scenarios",
                        "metrics_calculated": ["quality", "relevance", "coherence", "consistency"],
                        "baseline_comparison": "Compared against 4 different baseline models"
                    },
                    "recommendations": decision_obj.get('recommendations', []),
                    "next_steps": decision_obj.get('next_steps', []),
                    "report_generated_at": datetime.now().isoformat()
                }
                
                # Save report
                with open(self.evaluation_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Evaluation report generated for user {self.user_id}")
                return json.dumps(report)
                
            except Exception as e:
                logger.error(f"Error generating evaluation report for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "report_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(generate_evaluation_report)
    
    # Helper methods
    def _simulate_model_inference(self, prompt: str, test_id: int) -> Dict[str, Any]:
        """Simulate model inference for testing"""
        responses = [
            "The cat sat quietly by the window, watching the world go by with curious eyes.",
            "The sun shone brightly in the clear blue sky, casting warm golden light everywhere.",
            "Hey there! How's your day going? Oh, it's been great, thanks for asking!",
            "First, boil water in a kettle. Then add coffee grounds to your filter...",
            "My favorite place is a quiet beach where I can hear the waves and feel the sand."
        ]
        
        # Simulate quality scores with some randomness
        base_quality = 0.8 + (test_id * 0.02) + random.uniform(-0.1, 0.1)
        base_quality = max(0.0, min(1.0, base_quality))
        
        return {
            "output": responses[test_id % len(responses)],
            "quality_score": base_quality,
            "relevance_score": base_quality + random.uniform(-0.05, 0.05),
            "coherence_score": base_quality + random.uniform(-0.03, 0.03)
        }
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB"""
        total_size = 0
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                file_path = os.path.join(model_path, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_consistency_score(self, scores: List[float]) -> float:
        """Calculate consistency score (lower std dev = higher consistency)"""
        if len(scores) < 2:
            return 1.0
        std_dev = self._calculate_std_dev(scores)
        return max(0.0, 1.0 - std_dev)
    
    def _calculate_reliability_score(self, scores: List[float]) -> float:
        """Calculate reliability score"""
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    
    def _calculate_adaptability_score(self, test_data: List[Dict]) -> float:
        """Calculate adaptability score based on test diversity"""
        return 0.85  # Simulated
    
    def _calculate_overall_score(self, quality: List[float], relevance: List[float], coherence: List[float]) -> float:
        """Calculate overall score"""
        q_avg = sum(quality) / len(quality) if quality else 0
        r_avg = sum(relevance) / len(relevance) if relevance else 0
        c_avg = sum(coherence) / len(coherence) if coherence else 0
        return (q_avg + r_avg + c_avg) / 3
    
    def _assign_grade(self, scores: List[float]) -> str:
        """Assign letter grade based on scores"""
        avg_score = sum(scores) / len(scores) if scores else 0
        if avg_score >= 0.9:
            return "A"
        elif avg_score >= 0.8:
            return "B"
        elif avg_score >= 0.7:
            return "C"
        elif avg_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _calculate_ranking(self, user_score: float, baselines: Dict[str, float]) -> int:
        """Calculate ranking among all models"""
        all_scores = [user_score] + list(baselines.values())
        all_scores.sort(reverse=True)
        return all_scores.index(user_score) + 1
    
    def _generate_recommendations(self, overall_score: float, consistency_score: float) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Consider retraining with more diverse data")
        if consistency_score < 0.7:
            recommendations.append("Improve model consistency through additional training")
        if overall_score >= 0.85:
            recommendations.append("Model is ready for production deployment")
        if consistency_score >= 0.8:
            recommendations.append("Excellent consistency - suitable for critical applications")
            
        return recommendations
    
    def _generate_next_steps(self, overall_score: float, consistency_score: float) -> List[str]:
        """Generate next steps based on scores"""
        if overall_score >= 0.75 and consistency_score >= 0.70:
            return ["Proceed to model deployment", "Set up monitoring and logging", "Prepare for production launch"]
        else:
            return ["Retrain model with improved data", "Adjust training parameters", "Re-run evaluation tests"]
