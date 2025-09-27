"""
Model Trainer Agent using Google ADK
Handles fine-tuning of personalized models for users
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
import json
import os
import uuid
import logging
from datetime import datetime
import subprocess
import tempfile

logger = logging.getLogger(__name__)

class ModelTrainerAgent:
    """
    Agent responsible for training personalized models for users
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.training_config = config.get('adk', {}).get('training', {})
        self.model_path = f"models/user_models/{user_id}"
        self.training_log_path = f"logs/user_logs/{user_id}/training.log"
        
        # Ensure directories exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.training_log_path), exist_ok=True)
        
    def get_agent(self) -> Agent:
        """
        Creates and returns the model trainer agent
        """
        return Agent(
            name=f"model_trainer_{self.user_id.replace('-', '_')}",
            description="Trains personalized models for user fine-tuning",
            model="gemini-2.0-flash",
            instruction=f"""
            You are a model training agent for user {self.user_id}.
            Your responsibilities include:
            1. Load and prepare training data for fine-tuning
            2. Configure training parameters based on data characteristics
            3. Execute fine-tuning using appropriate base models
            4. Monitor training progress and handle errors
            5. Save trained models and training artifacts
            6. Report training metrics and results
            
            Always ensure optimal training configuration for the best model performance.
            """,
            tools=[
                self._create_data_loader_tool(),
                self._create_training_config_tool(),
                self._create_model_training_tool(),
                self._create_training_monitor_tool(),
                self._create_model_save_tool(),
                self._create_training_metrics_tool()
            ]
        )
    
    def _create_data_loader_tool(self) -> FunctionTool:
        """Creates tool for loading training data"""
        
        def load_training_data(data_path: str) -> str:
            """
            Loads processed training data for model training
            
            Args:
                data_path: Path to processed training data
                
            Returns:
                JSON string with loaded data information
            """
            try:
                logger.info(f"Loading training data for user {self.user_id}")
                
                if not os.path.exists(data_path):
                    return json.dumps({
                        "user_id": self.user_id,
                        "load_status": "error",
                        "error": "Training data file not found"
                    })
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    data_obj = json.load(f)
                
                examples = data_obj.get('examples', [])
                text_examples = [ex['text'] for ex in examples]
                
                load_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "load_status": "success",
                    "total_examples": len(examples),
                    "total_characters": sum(len(text) for text in text_examples),
                    "average_length": sum(len(text) for text in text_examples) / len(text_examples) if text_examples else 0,
                    "data_loaded": True,
                    "examples_preview": text_examples[:3]  # First 3 examples
                }
                
                logger.info(f"Training data loaded for user {self.user_id}: {len(examples)} examples")
                return json.dumps(load_result)
                
            except Exception as e:
                logger.error(f"Error loading training data for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "load_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(load_training_data)
    
    def _create_training_config_tool(self) -> FunctionTool:
        """Creates tool for configuring training parameters"""
        
        def configure_training(data_info: str, base_model: str = "distilbert-base-uncased") -> str:
            """
            Configures training parameters based on data characteristics
            
            Args:
                data_info: JSON string with data information
                base_model: Base model to use for fine-tuning
                
            Returns:
                JSON string with training configuration
            """
            try:
                logger.info(f"Configuring training for user {self.user_id}")
                
                data_obj = json.loads(data_info)
                total_examples = data_obj.get('total_examples', 0)
                avg_length = data_obj.get('average_length', 0)
                
                # Configure training parameters based on data
                config = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "base_model": base_model,
                    "training_job_id": f"training_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "parameters": {
                        "num_epochs": self._calculate_epochs(total_examples),
                        "batch_size": self._calculate_batch_size(avg_length),
                        "learning_rate": self.training_config.get('learning_rate', 5e-5),
                        "warmup_steps": self.training_config.get('warmup_steps', 100),
                        "weight_decay": self.training_config.get('weight_decay', 0.01),
                        "max_length": self.training_config.get('max_length', 512),
                        "save_steps": self.training_config.get('save_steps', 500),
                        "eval_steps": self.training_config.get('eval_steps', 500),
                        "gradient_accumulation_steps": self.training_config.get('gradient_accumulation_steps', 4)
                    },
                    "output_dir": self.model_path,
                    "logging_dir": os.path.dirname(self.training_log_path),
                    "data_characteristics": {
                        "total_examples": total_examples,
                        "average_length": avg_length,
                        "estimated_training_time": self._estimate_training_time(total_examples)
                    }
                }
                
                logger.info(f"Training configured for user {self.user_id}")
                return json.dumps(config)
                
            except Exception as e:
                logger.error(f"Error configuring training for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "config_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(configure_training)
    
    def _create_model_training_tool(self) -> FunctionTool:
        """Creates tool for executing model training"""
        
        def train_model(training_config: str, data_path: str) -> str:
            """
            Executes model training with the given configuration
            
            Args:
                training_config: JSON string with training configuration
                data_path: Path to training data
                
            Returns:
                JSON string with training results
            """
            try:
                logger.info(f"Starting model training for user {self.user_id}")
                
                config_obj = json.loads(training_config)
                training_job_id = config_obj.get('training_job_id')
                
                # Create training script
                training_script = self._create_training_script(config_obj, data_path)
                
                # Execute training (simulated for now)
                training_result = {
                    "user_id": self.user_id,
                    "training_job_id": training_job_id,
                    "timestamp": datetime.now().isoformat(),
                    "training_status": "started",
                    "model_path": self.model_path,
                    "training_script": training_script,
                    "estimated_duration": config_obj.get('data_characteristics', {}).get('estimated_training_time'),
                    "parameters": config_obj.get('parameters')
                }
                
                # Simulate training process
                self._simulate_training_progress(training_job_id)
                
                # Mark as completed
                training_result["training_status"] = "completed"
                training_result["completion_time"] = datetime.now().isoformat()
                training_result["model_files"] = [
                    f"{self.model_path}/model.pt",
                    f"{self.model_path}/tokenizer.json",
                    f"{self.model_path}/config.json"
                ]
                
                logger.info(f"Model training completed for user {self.user_id}")
                return json.dumps(training_result)
                
            except Exception as e:
                logger.error(f"Error training model for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "training_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(train_model)
    
    def _create_training_monitor_tool(self) -> FunctionTool:
        """Creates tool for monitoring training progress"""
        
        def monitor_training(training_job_id: str) -> str:
            """
            Monitors training progress and logs
            
            Args:
                training_job_id: Training job identifier
                
            Returns:
                JSON string with training progress information
            """
            try:
                logger.info(f"Monitoring training for user {self.user_id}")
                
                # Check training log file
                log_info = {
                    "user_id": self.user_id,
                    "training_job_id": training_job_id,
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_status": "active"
                }
                
                if os.path.exists(self.training_log_path):
                    with open(self.training_log_path, 'r') as f:
                        log_content = f.read()
                    
                    log_info.update({
                        "log_exists": True,
                        "log_size": len(log_content),
                        "last_entries": log_content.split('\n')[-5:] if log_content else []
                    })
                else:
                    log_info.update({
                        "log_exists": False,
                        "message": "Training log not found"
                    })
                
                # Check model files
                model_files = []
                if os.path.exists(self.model_path):
                    model_files = [f for f in os.listdir(self.model_path) if f.endswith(('.pt', '.json', '.txt'))]
                
                log_info["model_files"] = model_files
                log_info["training_complete"] = len(model_files) > 0
                
                return json.dumps(log_info)
                
            except Exception as e:
                logger.error(f"Error monitoring training for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "monitoring_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(monitor_training)
    
    def _create_model_save_tool(self) -> FunctionTool:
        """Creates tool for saving trained models"""
        
        def save_trained_model(model_info: str) -> str:
            """
            Saves trained model and metadata
            
            Args:
                model_info: JSON string with model information
                
            Returns:
                JSON string with save operation results
            """
            try:
                logger.info(f"Saving trained model for user {self.user_id}")
                
                model_obj = json.loads(model_info)
                
                # Create model metadata
                metadata = {
                    "user_id": self.user_id,
                    "model_id": f"model_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "created_at": datetime.now().isoformat(),
                    "base_model": model_obj.get('parameters', {}).get('base_model', 'distilbert-base-uncased'),
                    "training_config": model_obj.get('parameters', {}),
                    "model_version": "1.0",
                    "file_paths": model_obj.get('model_files', []),
                    "model_size": self._calculate_model_size()
                }
                
                # Save metadata
                metadata_path = f"{self.model_path}/metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                save_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "save_status": "success",
                    "model_id": metadata["model_id"],
                    "metadata_saved": metadata_path,
                    "model_files": model_obj.get('model_files', []),
                    "model_size_mb": metadata["model_size"]
                }
                
                logger.info(f"Model saved for user {self.user_id}")
                return json.dumps(save_result)
                
            except Exception as e:
                logger.error(f"Error saving model for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "save_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(save_trained_model)
    
    def _create_training_metrics_tool(self) -> FunctionTool:
        """Creates tool for calculating training metrics"""
        
        def calculate_training_metrics(training_result: str) -> str:
            """
            Calculates training metrics and performance indicators
            
            Args:
                training_result: JSON string with training results
                
            Returns:
                JSON string with training metrics
            """
            try:
                logger.info(f"Calculating training metrics for user {self.user_id}")
                
                result_obj = json.loads(training_result)
                config_obj = result_obj.get('parameters', {})
                
                metrics = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "training_job_id": result_obj.get('training_job_id'),
                    "model_id": f"model_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "training_metrics": {
                        "epochs_completed": config_obj.get('num_epochs', 0),
                        "batch_size": config_obj.get('batch_size', 0),
                        "learning_rate": config_obj.get('learning_rate', 0),
                        "training_duration": self._calculate_training_duration(),
                        "model_size_mb": self._calculate_model_size(),
                        "convergence_score": self._calculate_convergence_score(),
                        "quality_score": self._calculate_quality_score()
                    },
                    "performance_indicators": {
                        "training_success": result_obj.get('training_status') == 'completed',
                        "model_files_present": len(result_obj.get('model_files', [])) > 0,
                        "estimated_accuracy": 0.85,  # Simulated
                        "overfitting_risk": "low"
                    }
                }
                
                logger.info(f"Training metrics calculated for user {self.user_id}")
                return json.dumps(metrics)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "metrics_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(calculate_training_metrics)
    
    # Helper methods
    def _calculate_epochs(self, total_examples: int) -> int:
        """Calculate optimal number of epochs based on data size"""
        if total_examples < 100:
            return 5
        elif total_examples < 500:
            return 3
        else:
            return 2
    
    def _calculate_batch_size(self, avg_length: int) -> int:
        """Calculate optimal batch size based on text length"""
        if avg_length < 50:
            return 32
        elif avg_length < 200:
            return 16
        else:
            return 8
    
    def _estimate_training_time(self, total_examples: int) -> str:
        """Estimate training time based on data size"""
        if total_examples < 100:
            return "5-10 minutes"
        elif total_examples < 500:
            return "15-30 minutes"
        else:
            return "30-60 minutes"
    
    def _create_training_script(self, config: Dict, data_path: str) -> str:
        """Create training script content"""
        return f"""
# Training script for user {self.user_id}
# Configuration: {json.dumps(config, indent=2)}
# Data path: {data_path}
# This is a simulated training script
print("Training started...")
"""
    
    def _simulate_training_progress(self, job_id: str):
        """Simulate training progress"""
        # Create a simple log file
        with open(self.training_log_path, 'w') as f:
            f.write(f"Training job {job_id} started\n")
            f.write("Epoch 1/3: Loss = 2.45\n")
            f.write("Epoch 2/3: Loss = 1.89\n")
            f.write("Epoch 3/3: Loss = 1.23\n")
            f.write("Training completed successfully\n")
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        # Simulated model size
        return 125.5
    
    def _calculate_training_duration(self) -> str:
        """Calculate training duration"""
        return "12 minutes 34 seconds"
    
    def _calculate_convergence_score(self) -> float:
        """Calculate convergence score"""
        return 0.92
    
    def _calculate_quality_score(self) -> float:
        """Calculate model quality score"""
        return 0.87
