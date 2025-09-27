"""
Vertex AI Client for model management and deployment
Handles integration with Google Cloud Vertex AI services
"""

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import logging as cloud_logging
import json
import uuid
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class VertexAIClient:
    """
    Client for integration with Google Cloud Vertex AI
    Manages personalized models for each user
    """
    
    def __init__(self, project_id: str, location: str = "us-central1", config_path: str = "config/vertex_config.yaml"):
        self.project_id = project_id
        self.location = location
        self.config = self._load_config(config_path)
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location
        )
        
        # Initialize Cloud Storage
        self.storage_client = storage.Client(project=project_id)
        self.bucket_name = self.config.get('vertex_ai', {}).get('storage', {}).get('bucket_name', f"{project_id}-user-models")
        
        # Initialize Cloud Logging
        self.logging_client = cloud_logging.Client(project=project_id)
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        logger.info(f"Vertex AI Client initialized for project {project_id}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading Vertex AI config: {e}")
            return {}
    
    def _ensure_bucket_exists(self):
        """Ensure that the storage bucket exists"""
        try:
            self.storage_client.get_bucket(self.bucket_name)
            logger.info(f"Storage bucket {self.bucket_name} exists")
        except Exception:
            logger.info(f"Creating storage bucket {self.bucket_name}")
            bucket = self.storage_client.bucket(self.bucket_name)
            bucket.location = self.location
            self.storage_client.create_bucket(bucket)
    
    def create_custom_training_job(self, user_id: str, training_data: str, 
                                 base_model: str = "distilbert-base-uncased") -> str:
        """
        Creates a custom training job in Vertex AI
        
        Args:
            user_id: User identifier
            training_data: Training data for fine-tuning
            base_model: Base model to use for fine-tuning
            
        Returns:
            JSON string with training job information
        """
        try:
            logger.info(f"Creating custom training job for user {user_id}")
            
            job_id = f"training-{user_id}-{uuid.uuid4().hex[:8]}"
            
            # Upload training data to Cloud Storage
            blob_name = f"user-data/{user_id}/training_data.json"
            blob = self.storage_client.bucket(self.bucket_name).blob(blob_name)
            blob.upload_from_string(training_data)
            
            # Configure training job
            training_config = self.config.get('vertex_ai', {}).get('custom_training', {})
            
            job_config = {
                "display_name": f"User Model Training - {user_id}",
                "job_id": job_id,
                "user_id": user_id,
                "training_data_path": f"gs://{self.bucket_name}/{blob_name}",
                "base_model": base_model,
                "machine_type": training_config.get('machine_type', 'n1-standard-4'),
                "accelerator_type": training_config.get('accelerator_type', 'NVIDIA_TESLA_T4'),
                "accelerator_count": training_config.get('accelerator_count', 1),
                "boot_disk_type": training_config.get('boot_disk_type', 'pd-ssd'),
                "boot_disk_size_gb": training_config.get('boot_disk_size_gb', 100),
                "training_args": {
                    "num_epochs": 3,
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                    "max_length": 512
                },
                "created_at": datetime.now().isoformat(),
                "status": "PENDING"
            }
            
            # Here you would create the actual training job in Vertex AI
            # For now, we simulate the job creation
            self._simulate_training_job_creation(job_config)
            
            logger.info(f"Custom training job created for user {user_id}")
            return json.dumps(job_config)
            
        except Exception as e:
            logger.error(f"Error creating training job for user {user_id}: {e}")
            return json.dumps({
                "user_id": user_id,
                "job_creation_status": "error",
                "error": str(e)
            })
    
    def deploy_user_model(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """
        Deploys user model to Vertex AI Endpoints
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with deployment information
        """
        try:
            logger.info(f"Deploying model for user {user_id}")
            
            endpoint_name = f"user-endpoint-{user_id}"
            endpoint_display_name = f"User Model Endpoint - {user_id}"
            
            # Configure endpoint deployment
            deployment_config = self.config.get('vertex_ai', {}).get('model_deployment', {})
            
            endpoint_config = {
                "display_name": endpoint_display_name,
                "endpoint_id": endpoint_name,
                "user_id": user_id,
                "model_id": model_id,
                "region": self.location,
                "machine_type": deployment_config.get('machine_type', 'n1-standard-2'),
                "min_replica_count": deployment_config.get('min_replica_count', 1),
                "max_replica_count": deployment_config.get('max_replica_count', 10),
                "traffic_percentage": 100,
                "deployed_at": datetime.now().isoformat(),
                "endpoint_url": f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/endpoints/{endpoint_name}:predict"
            }
            
            # Here you would create the actual endpoint in Vertex AI
            # For now, we simulate the deployment
            self._simulate_endpoint_deployment(endpoint_config)
            
            logger.info(f"Model deployed for user {user_id}")
            return endpoint_config
            
        except Exception as e:
            logger.error(f"Error deploying model for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "deployment_status": "error",
                "error": str(e)
            }
    
    def make_inference(self, user_id: str, prompt: str, endpoint_url: str = None) -> str:
        """
        Makes inference using user's personalized model
        
        Args:
            user_id: User identifier
            prompt: Input prompt for inference
            endpoint_url: Endpoint URL (optional, will be looked up if not provided)
            
        Returns:
            Inference result
        """
        try:
            logger.info(f"Making inference for user {user_id}")
            
            if not endpoint_url:
                endpoint_url = self._get_user_endpoint_url(user_id)
            
            if not endpoint_url:
                return f"Endpoint not found for user {user_id}"
            
            # Prepare inference request
            inference_request = {
                "instances": [{
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.7
                }],
                "parameters": {
                    "temperature": 0.7,
                    "max_output_tokens": 100,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Here you would make the actual API call to Vertex AI
            # For now, we simulate the inference
            result = self._simulate_inference(prompt, user_id)
            
            # Log the inference
            self._log_inference(user_id, prompt, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making inference for user {user_id}: {e}")
            return f"Error during inference: {str(e)}"
    
    def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Gets the status of a training job
        
        Args:
            job_id: Training job identifier
            
        Returns:
            Dictionary with job status information
        """
        try:
            logger.info(f"Getting training job status for {job_id}")
            
            # Here you would query the actual training job status
            # For now, we simulate the status
            status = self._simulate_training_job_status(job_id)
            
            return {
                "job_id": job_id,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "progress_percentage": self._get_training_progress(job_id),
                "estimated_completion": self._get_estimated_completion(job_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting training job status for {job_id}: {e}")
            return {
                "job_id": job_id,
                "status": "ERROR",
                "error": str(e)
            }
    
    def list_user_models(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Lists all models for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of model information dictionaries
        """
        try:
            logger.info(f"Listing models for user {user_id}")
            
            # Query models for the user
            models = []
            
            # Check for model files in storage
            prefix = f"user-models/{user_id}/"
            blobs = self.storage_client.list_blobs(self.bucket_name, prefix=prefix)
            
            for blob in blobs:
                if blob.name.endswith('.pt') or blob.name.endswith('.json'):
                    model_info = {
                        "model_id": blob.name.split('/')[-1],
                        "user_id": user_id,
                        "file_path": f"gs://{self.bucket_name}/{blob.name}",
                        "size_bytes": blob.size,
                        "created_at": blob.time_created.isoformat(),
                        "updated_at": blob.updated.isoformat()
                    }
                    models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models for user {user_id}: {e}")
            return []
    
    def delete_user_model(self, user_id: str, model_id: str) -> bool:
        """
        Deletes a user's model
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            logger.info(f"Deleting model {model_id} for user {user_id}")
            
            # Delete model files from storage
            prefix = f"user-models/{user_id}/{model_id}"
            blobs = self.storage_client.list_blobs(self.bucket_name, prefix=prefix)
            
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} files for model {model_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id} for user {user_id}: {e}")
            return False
    
    def setup_monitoring(self, user_id: str, endpoint_id: str) -> Dict[str, Any]:
        """
        Sets up monitoring for a user's deployed model
        
        Args:
            user_id: User identifier
            endpoint_id: Endpoint identifier
            
        Returns:
            Dictionary with monitoring configuration
        """
        try:
            logger.info(f"Setting up monitoring for user {user_id}")
            
            monitoring_config = self.config.get('vertex_ai', {}).get('monitoring', {})
            
            config = {
                "user_id": user_id,
                "endpoint_id": endpoint_id,
                "monitoring_enabled": monitoring_config.get('enable_monitoring', True),
                "metrics_export_interval": monitoring_config.get('metrics_export_interval', '60s'),
                "log_sampling_rate": monitoring_config.get('log_sampling_rate', 1.0),
                "alerting_config": {
                    "cpu_threshold": 80,
                    "memory_threshold": 85,
                    "response_time_threshold": 5.0,
                    "error_rate_threshold": 0.05
                },
                "dashboard_url": f"https://console.cloud.google.com/monitoring/dashboards/endpoint/{endpoint_id}",
                "created_at": datetime.now().isoformat()
            }
            
            # Here you would set up actual monitoring in Google Cloud
            # For now, we simulate the setup
            self._simulate_monitoring_setup(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error setting up monitoring for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "monitoring_status": "error",
                "error": str(e)
            }
    
    def get_model_metrics(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """
        Gets metrics for a deployed model
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with model metrics
        """
        try:
            logger.info(f"Getting metrics for model {model_id} of user {user_id}")
            
            # Here you would query actual metrics from Google Cloud Monitoring
            # For now, we simulate the metrics
            metrics = self._simulate_model_metrics(user_id, model_id)
            
            return {
                "user_id": user_id,
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics for model {model_id}: {e}")
            return {
                "user_id": user_id,
                "model_id": model_id,
                "metrics_status": "error",
                "error": str(e)
            }
    
    # Helper methods
    def _get_user_endpoint_url(self, user_id: str) -> Optional[str]:
        """Get endpoint URL for a user"""
        # In a real implementation, you would query the endpoint registry
        # For now, we simulate
        return f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/endpoints/user-endpoint-{user_id}:predict"
    
    def _simulate_training_job_creation(self, job_config: Dict[str, Any]):
        """Simulate training job creation"""
        # In a real implementation, you would create the job in Vertex AI
        logger.info(f"Simulated training job creation: {job_config['job_id']}")
    
    def _simulate_endpoint_deployment(self, endpoint_config: Dict[str, Any]):
        """Simulate endpoint deployment"""
        # In a real implementation, you would deploy to Vertex AI
        logger.info(f"Simulated endpoint deployment: {endpoint_config['endpoint_id']}")
    
    def _simulate_inference(self, prompt: str, user_id: str) -> str:
        """Simulate model inference"""
        # In a real implementation, you would call the actual endpoint
        responses = [
            f"Response from user {user_id} model: I understand you're asking about '{prompt}'. Let me provide a helpful response.",
            f"Based on your personalized model for user {user_id}, here's what I think about '{prompt}'...",
            f"Your custom model for user {user_id} suggests that '{prompt}' is an interesting topic. Here's my analysis..."
        ]
        
        import random
        return random.choice(responses)
    
    def _log_inference(self, user_id: str, prompt: str, result: str):
        """Log inference request and response"""
        try:
            log_entry = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": result,
                "response_length": len(result)
            }
            
            # Here you would log to Cloud Logging
            # For now, we simulate logging
            logger.info(f"Inference logged for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error logging inference for user {user_id}: {e}")
    
    def _simulate_training_job_status(self, job_id: str) -> str:
        """Simulate training job status"""
        statuses = ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
        import random
        return random.choice(statuses)
    
    def _get_training_progress(self, job_id: str) -> int:
        """Get training progress percentage"""
        import random
        return random.randint(0, 100)
    
    def _get_estimated_completion(self, job_id: str) -> str:
        """Get estimated completion time"""
        import random
        hours = random.randint(1, 6)
        minutes = random.randint(0, 59)
        return f"{hours}h {minutes}m"
    
    def _simulate_monitoring_setup(self, config: Dict[str, Any]):
        """Simulate monitoring setup"""
        logger.info(f"Simulated monitoring setup for user {config['user_id']}")
    
    def _simulate_model_metrics(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """Simulate model metrics"""
        import random
        
        return {
            "requests_per_minute": random.randint(10, 100),
            "average_response_time": random.uniform(0.5, 3.0),
            "error_rate": random.uniform(0.0, 0.05),
            "cpu_utilization": random.uniform(20, 80),
            "memory_utilization": random.uniform(30, 90),
            "active_replicas": random.randint(1, 5),
            "total_requests": random.randint(100, 10000),
            "success_rate": random.uniform(0.95, 1.0)
        }
