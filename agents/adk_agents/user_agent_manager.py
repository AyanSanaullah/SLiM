"""
User Agent Manager using Google ADK
Manages personalized agents for each user with individual model fine-tuning
"""

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.sessions import Session
from typing import Dict, List, Optional, Any
import json
import uuid
import os
import yaml
import logging
from datetime import datetime
import asyncio
import threading

from .data_processor_agent import DataProcessorAgent
from .model_trainer_agent import ModelTrainerAgent
from .model_evaluator_agent import ModelEvaluatorAgent
from .model_deployer_agent import ModelDeployerAgent
from .real_model_trainer import RealModelTrainer
from .advanced_training_agent import AdvancedTrainingAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserAgentManager:
    """
    Manages personalized agents for each user using Google ADK
    Each user gets their own agent pipeline with custom model fine-tuning
    """
    
    def __init__(self, config_path: str = "config/adk_config.yaml"):
        self.active_users: Dict[str, Dict] = {}
        self.user_agents: Dict[str, Agent] = {}
        self.user_pipelines: Dict[str, SequentialAgent] = {}
        self.config = self._load_config(config_path)
        self.vertex_client = None  # Will be initialized with credentials
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("UserAgentManager initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _create_directories(self):
        """Create necessary directories for user data and models"""
        directories = [
            self.config.get('adk', {}).get('storage', {}).get('user_data_path', 'data/user_data'),
            self.config.get('adk', {}).get('storage', {}).get('user_models_path', 'models/user_models'),
            self.config.get('adk', {}).get('storage', {}).get('user_logs_path', 'logs/user_logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_user_agent_pipeline(self, user_id: str, training_data: str, 
                                  base_model: str = "distilbert-base-uncased") -> str:
        """
        Creates a personalized agent pipeline for a user using ADK
        
        Args:
            user_id: Unique identifier for the user
            training_data: Training data for fine-tuning
            base_model: Base model to use for fine-tuning
            
        Returns:
            Status message about pipeline creation
        """
        try:
            logger.info(f"Creating agent pipeline for user: {user_id}")
            
            # Create user pipeline using ADK
            user_pipeline = self._create_user_pipeline(user_id, training_data, base_model)
            
            # Register user
            self.active_users[user_id] = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'status': 'initializing',
                'pipeline_id': str(uuid.uuid4()),
                'base_model': base_model,
                'training_data_size': len(training_data)
            }
            
            # Store pipeline
            self.user_pipelines[user_id] = user_pipeline
            
            # Execute pipeline asynchronously
            session = Session(
                id=f"session_{user_id}_{uuid.uuid4().hex[:8]}",
                app_name="shellhacks-adk-agents",
                user_id=user_id,
                state={
                    "user_id": user_id,
                    "training_data": training_data,
                    "base_model": base_model
                }
            )
            
            # Start pipeline execution in separate thread
            pipeline_thread = threading.Thread(
                target=self._execute_pipeline,
                args=(user_id, session)
            )
            pipeline_thread.start()
            
            return f"Agent pipeline created and started for user {user_id}"
            
        except Exception as e:
            logger.error(f"Error creating user agent pipeline: {e}")
            if user_id in self.active_users:
                self.active_users[user_id]['status'] = 'error'
                self.active_users[user_id]['error'] = str(e)
            return f"Error creating user agent: {str(e)}"
    
    def create_advanced_agent_pipeline(self, user_id: str, json_dataset: List[Dict[str, str]], 
                                     base_model: str = "distilbert-base-uncased") -> str:
        """
        Creates an advanced agent pipeline for JSON dataset training with string comparison
        
        Args:
            user_id: Unique identifier for the user
            json_dataset: List of {"prompt": "...", "answer": "..."} dictionaries
            base_model: Base model to use for fine-tuning
            
        Returns:
            Status message about pipeline creation
        """
        try:
            logger.info(f"Creating ADVANCED agent pipeline for user: {user_id}")
            
            # Initialize advanced training agent
            advanced_agent = AdvancedTrainingAgent(user_id, self.config)
            
            # Register user
            self.active_users[user_id] = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'status': 'initializing',
                'pipeline_id': str(uuid.uuid4()),
                'base_model': base_model,
                'training_type': 'advanced_json_qlora',
                'dataset_size': len(json_dataset),
                'string_comparison_enabled': True
            }
            
            # Store advanced agent
            self.user_model_trainers = getattr(self, 'user_model_trainers', {})
            self.user_model_trainers[user_id] = advanced_agent
            
            # Execute advanced pipeline asynchronously
            pipeline_thread = threading.Thread(
                target=self._execute_advanced_pipeline,
                args=(user_id, json_dataset, base_model)
            )
            pipeline_thread.start()
            
            return f"Advanced agent pipeline created for user {user_id} with {len(json_dataset)} samples"
            
        except Exception as e:
            logger.error(f"Error creating advanced agent pipeline: {e}")
            if user_id in self.active_users:
                self.active_users[user_id]['status'] = 'error'
                self.active_users[user_id]['error'] = str(e)
            return f"Error creating advanced agent: {str(e)}"
    
    def _create_user_pipeline(self, user_id: str, training_data: str, base_model: str) -> SequentialAgent:
        """
        Creates a sequential pipeline of agents for a user
        
        Pipeline: Data Processor -> Model Trainer -> Model Evaluator -> Model Deployer
        """
        
        # Create individual agents
        data_processor = DataProcessorAgent(user_id, self.config)
        model_trainer = ModelTrainerAgent(user_id, self.config)
        model_evaluator = ModelEvaluatorAgent(user_id, self.config)
        model_deployer = ModelDeployerAgent(user_id, self.config)
        
        # Create sequential pipeline
        pipeline = SequentialAgent(
            name=f"user_pipeline_{user_id.replace('-', '_')}",
            sub_agents=[
                data_processor.get_agent(),
                model_trainer.get_agent(),
                model_evaluator.get_agent(),
                model_deployer.get_agent()
            ]
        )
        
        return pipeline
    
    def _execute_pipeline(self, user_id: str, session: Session):
        """Execute user pipeline in separate thread - REAL MODEL TRAINING"""
        try:
            self.active_users[user_id]['status'] = 'processing'
            self.active_users[user_id]['training_progress'] = 0
            
            # Get training data and base model from session
            training_data = session.state.get('training_data', '')
            base_model = session.state.get('base_model', 'distilbert-base-uncased')
            
            # Initialize real model trainer
            model_trainer = RealModelTrainer(user_id, self.config)
            
            # Step 1: Process training data into prompt/answer pairs (20% progress)
            logger.info(f"Processing training data for user {user_id}")
            self.active_users[user_id]['training_progress'] = 20
            processed_data = model_trainer.process_training_data(training_data)
            
            # Step 2: Train the actual model (60% progress)
            logger.info(f"Training model for user {user_id}")
            self.active_users[user_id]['training_progress'] = 60
            training_results = model_trainer.train_model(processed_data)
            
            # Step 3: Test the model with a sample evaluation (90% progress)
            logger.info(f"Testing model for user {user_id}")
            self.active_users[user_id]['training_progress'] = 90
            test_results = model_trainer.evaluate_model(
                "How can you help?",
                "I can help with questions and explanations."
            )
            
            # Update user status (without storing the model_trainer object for JSON serialization)
            self.active_users[user_id]['status'] = 'ready'
            self.active_users[user_id]['training_progress'] = 60
            self.active_users[user_id]['model_ready_at'] = datetime.now().isoformat()
            self.active_users[user_id]['initial_training_complete'] = True
            self.active_users[user_id]['continuous_training'] = False  # Ready for continuous training
            self.active_users[user_id]['processed_data'] = processed_data
            self.active_users[user_id]['training_results'] = training_results
            self.active_users[user_id]['test_results'] = test_results
            
            # Store model trainer separately (not serializable)
            self.user_model_trainers = getattr(self, 'user_model_trainers', {})
            self.user_model_trainers[user_id] = model_trainer
            
            # Log performance results
            self._log_training_performance(user_id, {
                'training_results': training_results,
                'test_results': test_results,
                'processed_data': processed_data,
                'agent_type': 'real_model'
            })
            
            logger.info(f"Real model training completed for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error executing real training pipeline for user {user_id}: {e}")
            self.active_users[user_id]['status'] = 'error'
            self.active_users[user_id]['training_progress'] = 0
            self.active_users[user_id]['error'] = str(e)
    
    def _execute_advanced_pipeline(self, user_id: str, json_dataset: List[Dict[str, str]], base_model: str):
        """Execute advanced pipeline with JSON dataset and string comparison"""
        try:
            self.active_users[user_id]['status'] = 'processing'
            self.active_users[user_id]['training_progress'] = 0
            
            # Get the advanced training agent
            advanced_agent = self.user_model_trainers[user_id]
            
            # Step 1: Process JSON dataset (25% progress)
            logger.info(f"Processing JSON dataset for user {user_id}")
            self.active_users[user_id]['training_progress'] = 25
            dataset_stats = advanced_agent.process_json_dataset(json_dataset)
            
            # Step 2: Train with QLoRA (70% progress)
            logger.info(f"Training with QLoRA for user {user_id}")
            self.active_users[user_id]['training_progress'] = 70
            training_results = advanced_agent.train_with_qlora(dataset_stats)
            
            # Step 3: Evaluate full dataset with string comparison (95% progress)
            logger.info(f"Evaluating with string comparison for user {user_id}")
            self.active_users[user_id]['training_progress'] = 95
            evaluation_results = advanced_agent.evaluate_full_dataset()
            
            # Update user status
            self.active_users[user_id]['status'] = 'ready'
            self.active_users[user_id]['training_progress'] = 100
            self.active_users[user_id]['model_ready_at'] = datetime.now().isoformat()
            self.active_users[user_id]['dataset_stats'] = dataset_stats
            self.active_users[user_id]['training_results'] = training_results
            self.active_users[user_id]['evaluation_results'] = evaluation_results
            
            # Log advanced performance results with detailed string comparison
            self._log_training_performance(user_id, {
                'dataset_stats': dataset_stats,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'agent_type': 'advanced_qlora'
            })
            
            logger.info(f"Advanced pipeline completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error executing advanced pipeline for user {user_id}: {e}")
            self.active_users[user_id]['status'] = 'error'
            self.active_users[user_id]['training_progress'] = 0
            self.active_users[user_id]['error'] = str(e)
    
    def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a user's agent
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user status information or None if not found
        """
        if user_id not in self.active_users:
            return None
        
        user_info = self.active_users[user_id].copy()
        
        # Add additional information if model is ready
        if user_info['status'] == 'ready':
            model_path = f"models/user_models/{user_id}"
            if os.path.exists(f"{model_path}/model.pt"):
                user_info['model_exists'] = True
                user_info['model_size'] = os.path.getsize(f"{model_path}/model.pt")
            else:
                user_info['model_exists'] = False
        
        return user_info
    
    def make_inference(self, user_id: str, prompt: str) -> str:
        """
        Make inference using user's TRAINED model
        
        Args:
            user_id: User identifier
            prompt: Input prompt for inference
            
        Returns:
            Inference result or error message
        """
        if user_id not in self.active_users:
            return f"User {user_id} not found"
        
        if self.active_users[user_id]['status'] != 'ready':
            return f"Model for user {user_id} is not ready yet (status: {self.active_users[user_id]['status']})"
        
        try:
            # Get the trained model trainer from separate storage
            self.user_model_trainers = getattr(self, 'user_model_trainers', {})
            model_trainer = self.user_model_trainers.get(user_id)
            
            if not model_trainer:
                # Recreate model trainer if not in memory
                model_trainer = RealModelTrainer(user_id, self.config)
                self.user_model_trainers[user_id] = model_trainer
            
            # Check if it's an advanced agent
            if isinstance(model_trainer, AdvancedTrainingAgent):
                # Use advanced inference with string comparison
                result = model_trainer.make_inference_and_compare(prompt)
                
                if 'error' in result:
                    return f"Error: {result['error']}"
                
                model_response = result['model_response']
                confidence = result['model_confidence']
                string_comp = result.get('string_comparison', {})
                # Use semantic_similarity if available, fallback to similarity
                similarity = string_comp.get('semantic_similarity', string_comp.get('similarity', 0.0))
                
                response = f"{model_response}\n\n[Confiança do Modelo: {confidence:.2%}]\n[Similaridade String: {similarity:.2%}]"
                
                logger.info(f"Advanced model inference for user {user_id}: confidence={confidence:.3f}, similarity={similarity:.3f}")
                
                return response
            else:
                # Use regular inference
                answer, confidence = model_trainer.make_inference(prompt)
                
                # Format response with confidence
                response = f"{answer}\n\n[Confiança: {confidence:.2%}]"
                
                logger.info(f"Real model inference for user {user_id}: prompt='{prompt}', confidence={confidence:.3f}")
                
                return response
                
        except Exception as e:
            logger.error(f"Error during real model inference for user {user_id}: {e}")
            return f"Error during inference: {str(e)}"
    
    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active users
        
        Returns:
            Dictionary with all active users and their information
        """
        return self.active_users.copy()
    
    def _process_training_data(self, user_id: str, training_data: str) -> Dict[str, Any]:
        """Process training data for the user"""
        try:
            # Create data processor agent
            data_processor = DataProcessorAgent(user_id, self.config)
            
            # Process the data
            processed_data = {
                "user_id": user_id,
                "original_size": len(training_data),
                "processed_at": datetime.now().isoformat(),
                "data_preview": training_data[:200] + "..." if len(training_data) > 200 else training_data
            }
            
            logger.info(f"Data processed for user {user_id}: {processed_data['original_size']} characters")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data for user {user_id}: {e}")
            return {"error": str(e)}
    
    def _create_vertex_endpoint(self, user_id: str, base_model: str) -> Optional[str]:
        """Create a Vertex AI endpoint for the user"""
        try:
            # For now, return a placeholder endpoint
            # In a real implementation, you would create an actual Vertex AI endpoint
            endpoint_id = f"endpoint-{user_id}-{uuid.uuid4().hex[:8]}"
            endpoint_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.config.get('project_id', 'your-project')}/locations/us-central1/endpoints/{endpoint_id}"
            
            logger.info(f"Created endpoint for user {user_id}: {endpoint_id}")
            return endpoint_url
            
        except Exception as e:
            logger.error(f"Error creating endpoint for user {user_id}: {e}")
            return None
    
    def evaluate_model_performance(self, user_id: str, test_prompt: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate model performance using string comparison
        
        Args:
            user_id: User identifier
            test_prompt: Test prompt
            expected_answer: Expected answer
            
        Returns:
            Evaluation results with similarity metrics
        """
        try:
            if user_id not in self.active_users:
                return {"error": f"User {user_id} not found"}
            
            self.user_model_trainers = getattr(self, 'user_model_trainers', {})
            model_trainer = self.user_model_trainers.get(user_id)
            if not model_trainer:
                model_trainer = RealModelTrainer(user_id, self.config)
                self.user_model_trainers[user_id] = model_trainer
            
            # Check if it's an advanced agent and use string comparison service
            if isinstance(model_trainer, AdvancedTrainingAgent):
                # Use advanced inference with string comparison for evaluation
                inference_result = model_trainer.make_inference_and_compare(test_prompt)
                
                if 'error' in inference_result:
                    return {"error": inference_result['error']}
                
                string_comp = inference_result.get('string_comparison', {})
                
                evaluation_results = {
                    'test_prompt': test_prompt,
                    'expected_answer': expected_answer,
                    'predicted_answer': inference_result['model_response'],
                    'model_confidence': inference_result['model_confidence'],
                    'semantic_similarity': string_comp.get('semantic_similarity', string_comp.get('similarity', 0.0)),
                    'overall_similarity': string_comp.get('semantic_similarity', string_comp.get('similarity', 0.0)),
                    'string_comparison_details': string_comp,
                    'evaluated_at': datetime.now().isoformat()
                }
            else:
                # Use regular model trainer evaluation
                evaluation_results = model_trainer.evaluate_model(test_prompt, expected_answer)
            
            logger.info(f"Model evaluation for user {user_id}: {evaluation_results.get('overall_similarity', 0):.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model for user {user_id}: {e}")
            return {"error": str(e)}
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user and clean up their resources
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user was removed, False if not found
        """
        if user_id in self.active_users:
            # Clean up resources
            try:
                # Remove model files
                model_path = f"models/user_models/{user_id}"
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                
                # Remove data files
                data_path = f"data/user_data/{user_id}"
                if os.path.exists(data_path):
                    shutil.rmtree(data_path)
                
                # Remove from active users
                del self.active_users[user_id]
                
                # Remove pipeline
                if user_id in self.user_pipelines:
                    del self.user_pipelines[user_id]
                
                # Remove model trainer
                self.user_model_trainers = getattr(self, 'user_model_trainers', {})
                if user_id in self.user_model_trainers:
                    del self.user_model_trainers[user_id]
                
                logger.info(f"User {user_id} removed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error removing user {user_id}: {e}")
                return False
        
        return False
    
    def remove_all_users(self) -> Dict[str, Any]:
        """
        Remove all users and clean up their resources
        
        Returns:
            Dictionary with removal results
        """
        result = {
            'total_users': len(self.active_users),
            'removed_users': [],
            'failed_users': [],
            'success': True
        }
        
        # Get list of all user IDs to avoid modifying dict during iteration
        user_ids = list(self.active_users.keys())
        
        logger.info(f"Starting removal of {len(user_ids)} users")
        
        for user_id in user_ids:
            try:
                if self.remove_user(user_id):
                    result['removed_users'].append(user_id)
                    logger.info(f"Successfully removed user: {user_id}")
                else:
                    result['failed_users'].append(user_id)
                    logger.warning(f"Failed to remove user: {user_id}")
            except Exception as e:
                result['failed_users'].append(user_id)
                logger.error(f"Exception while removing user {user_id}: {e}")
        
        # Clean up any remaining data directories
        try:
            import shutil
            
            # Clean up models directory
            models_dir = "models/user_models"
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
                os.makedirs(models_dir, exist_ok=True)
            
            # Clean up data directory
            data_dir = "data/user_data"
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                os.makedirs(data_dir, exist_ok=True)
                
            logger.info("Cleaned up remaining user directories")
            
        except Exception as e:
            logger.error(f"Error cleaning up directories: {e}")
        
        # Clear all collections
        self.active_users.clear()
        self.user_pipelines.clear()
        if hasattr(self, 'user_model_trainers'):
            self.user_model_trainers.clear()
        
        result['success'] = len(result['failed_users']) == 0
        
        logger.info(f"Removal completed. Success: {result['success']}, "
                   f"Removed: {len(result['removed_users'])}, "
                   f"Failed: {len(result['failed_users'])}")
        
        return result
    
    def get_pipeline_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of user's pipeline
        
        Args:
            user_id: User identifier
            
        Returns:
            Pipeline status information or None if not found
        """
        if user_id not in self.active_users:
            return None
        
        pipeline_info = {
            'user_id': user_id,
            'status': self.active_users[user_id]['status'],
            'created_at': self.active_users[user_id]['created_at'],
            'pipeline_id': self.active_users[user_id]['pipeline_id'],
            'base_model': self.active_users[user_id].get('base_model'),
            'training_data_size': self.active_users[user_id].get('training_data_size')
        }
        
        # Add model ready time if available
        if 'model_ready_at' in self.active_users[user_id]:
            pipeline_info['model_ready_at'] = self.active_users[user_id]['model_ready_at']
        
        # Add error information if available
        if 'error' in self.active_users[user_id]:
            pipeline_info['error'] = self.active_users[user_id]['error']
        
        return pipeline_info
    
    def _log_training_performance(self, user_id: str, performance_data: Dict[str, Any]):
        """
        Log detailed training performance with string comparison results
        
        Args:
            user_id: User identifier
            performance_data: Performance metrics and results
        """
        try:
            agent_type = performance_data.get('agent_type', 'unknown')
            
            # Print header
            print("\n" + "🎯" * 40)
            print(f"🎯 AGENT PERFORMANCE REPORT - {user_id.upper()}")
            print("🎯" * 40)
            print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🤖 Agent Type: {agent_type}")
            
            if agent_type == 'advanced_qlora':
                self._log_advanced_performance(user_id, performance_data)
            else:
                self._log_basic_performance(user_id, performance_data)
            
            print("🎯" * 40)
            print("")
            
        except Exception as e:
            logger.error(f"Error logging performance for {user_id}: {e}")
    
    def _log_advanced_performance(self, user_id: str, performance_data: Dict[str, Any]):
        """Log advanced agent performance with detailed string comparison"""
        
        dataset_stats = performance_data.get('dataset_stats', {})
        training_results = performance_data.get('training_results', {})
        evaluation_results = performance_data.get('evaluation_results', {})
        
        # Dataset info
        print(f"📊 DATASET STATISTICS")
        print(f"   Total Samples: {dataset_stats.get('total_samples', 0)}")
        print(f"   Valid Samples: {dataset_stats.get('valid_samples', 0)}")
        print(f"   Average Prompt Length: {dataset_stats.get('avg_prompt_length', 0):.1f} chars")
        print(f"   Average Answer Length: {dataset_stats.get('avg_answer_length', 0):.1f} chars")
        
        # Training info
        print(f"\n🚀 TRAINING RESULTS")
        print(f"   Model Type: {training_results.get('model_type', 'unknown')}")
        print(f"   Device: {training_results.get('device', 'cpu')}")
        print(f"   CUDA Available: {'✅' if training_results.get('cuda_available') else '❌'}")
        print(f"   GPU Count: {training_results.get('gpu_count', 0)}")
        print(f"   Training Time: {training_results.get('training_time_minutes', 0):.2f} minutes")
        print(f"   QLoRA Config: rank={training_results.get('lora_rank', 0)}, alpha={training_results.get('lora_alpha', 0)}")
        
        # GPU info
        if 'gpu_info' in training_results:
            print(f"   GPU Details:")
            for gpu in training_results['gpu_info']:
                print(f"     - GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        
        # String comparison evaluation
        print(f"\n🔍 STRING COMPARISON EVALUATION")
        total_samples = evaluation_results.get('total_samples', 0)
        successful = evaluation_results.get('successful_evaluations', 0)
        failed = evaluation_results.get('failed_evaluations', 0)
        
        print(f"   Total Tests: {total_samples}")
        print(f"   Successful: {successful} ({successful/total_samples*100:.1f}%)" if total_samples > 0 else "   Successful: 0 (0%)")
        print(f"   Failed: {failed} ({failed/total_samples*100:.1f}%)" if total_samples > 0 else "   Failed: 0 (0%)")
        
        # Similarity metrics
        avg_similarity = evaluation_results.get('average_string_similarity', 0)
        avg_confidence = evaluation_results.get('average_model_confidence', 0)
        max_similarity = evaluation_results.get('max_string_similarity', 0)
        min_similarity = evaluation_results.get('min_string_similarity', 0)
        
        print(f"\n📈 SIMILARITY METRICS")
        print(f"   Average String Similarity: {avg_similarity:.2%}")
        print(f"   Average Model Confidence: {avg_confidence:.2%}")
        print(f"   Best Similarity: {max_similarity:.2%}")
        print(f"   Worst Similarity: {min_similarity:.2%}")
        
        # Performance categories
        high_count = evaluation_results.get('high_similarity_count', 0)
        medium_count = evaluation_results.get('medium_similarity_count', 0)
        low_count = evaluation_results.get('low_similarity_count', 0)
        
        print(f"\n🎯 PERFORMANCE DISTRIBUTION")
        if total_samples > 0:
            print(f"   🟢 High Quality (>80%): {high_count} ({high_count/total_samples*100:.1f}%)")
            print(f"   🟡 Medium Quality (50-80%): {medium_count} ({medium_count/total_samples*100:.1f}%)")
            print(f"   🔴 Low Quality (<50%): {low_count} ({low_count/total_samples*100:.1f}%)")
        else:
            print(f"   No evaluation data available")
        
        # Overall score
        overall_score = self._calculate_overall_score(evaluation_results)
        print(f"\n🏆 OVERALL PERFORMANCE SCORE: {overall_score:.1%}")
        
        # Performance emoji
        if overall_score >= 0.8:
            print("🏆 EXCELLENT PERFORMANCE!")
        elif overall_score >= 0.6:
            print("👍 GOOD PERFORMANCE!")
        elif overall_score >= 0.4:
            print("📈 FAIR PERFORMANCE - Need Improvement")
        else:
            print("⚠️ POOR PERFORMANCE - Requires Attention")
    
    def _log_basic_performance(self, user_id: str, performance_data: Dict[str, Any]):
        """Log basic agent performance"""
        
        training_results = performance_data.get('training_results', {})
        test_results = performance_data.get('test_results', {})
        processed_data = performance_data.get('processed_data', {})
        
        # Dataset info
        print(f"📊 DATASET STATISTICS")
        print(f"   Original Data Size: {processed_data.get('original_size', 0)} chars")
        print(f"   Training Pairs: {training_results.get('training_pairs_count', 0)}")
        
        # Training info
        print(f"\n🚀 TRAINING RESULTS")
        print(f"   Model Type: {training_results.get('model_type', 'unknown')}")
        print(f"   Status: {training_results.get('status', 'unknown')}")
        
        # Test results
        print(f"\n🔍 EVALUATION RESULTS")
        print(f"   Test Prompt: {test_results.get('test_prompt', 'N/A')}")
        print(f"   Model Confidence: {test_results.get('model_confidence', 0):.2%}")
        
        # Similarity metrics
        jaccard = test_results.get('jaccard_similarity', 0)
        character = test_results.get('character_similarity', 0)
        overall = test_results.get('overall_similarity', 0)
        
        print(f"\n📈 SIMILARITY METRICS")
        print(f"   Jaccard Similarity: {jaccard:.2%}")
        print(f"   Character Similarity: {character:.2%}")
        print(f"   Overall Similarity: {overall:.2%}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT")
        if overall >= 0.8:
            print("🏆 EXCELLENT - High quality match!")
        elif overall >= 0.6:
            print("👍 GOOD - Satisfactory performance")
        elif overall >= 0.4:
            print("📈 FAIR - Moderate performance")
        else:
            print("⚠️ POOR - Low quality match")
    
    def _calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        try:
            if not evaluation_results:
                return 0.0
            
            # Base score on average string similarity
            avg_similarity = evaluation_results.get('average_string_similarity', 0)
            
            # Success rate bonus
            total_samples = evaluation_results.get('total_samples', 1)
            successful = evaluation_results.get('successful_evaluations', 0)
            success_rate = successful / total_samples if total_samples > 0 else 0
            
            # High quality rate bonus
            high_count = evaluation_results.get('high_similarity_count', 0)
            high_rate = high_count / total_samples if total_samples > 0 else 0
            
            # Weighted score
            score = (
                avg_similarity * 0.6 +    # 60% weight on average similarity
                success_rate * 0.25 +     # 25% weight on success rate
                high_rate * 0.15          # 15% weight on high quality rate
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0


# Global instance
user_agent_manager = UserAgentManager()
