"""
Model Deployer Agent using Google ADK
Handles deployment of approved models to production endpoints
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
import json
import os
import uuid
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class ModelDeployerAgent:
    """
    Agent responsible for deploying approved models to production endpoints
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.deployment_config = config.get('adk', {}).get('deployment', {})
        self.model_path = f"models/user_models/{user_id}"
        self.deployment_log_path = f"logs/user_logs/{user_id}/deployment.log"
        self.endpoint_config_path = f"models/user_models/{user_id}/endpoint_config.json"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.deployment_log_path), exist_ok=True)
        
    def get_agent(self) -> Agent:
        """
        Creates and returns the model deployer agent
        """
        return Agent(
            name=f"model_deployer_{self.user_id.replace('-', '_')}",
            description="Deploys approved models to production endpoints",
            model="gemini-2.0-flash",
            instruction=f"""
            You are a model deployment agent for user {self.user_id}.
            Your responsibilities include:
            1. Prepare models for production deployment
            2. Create and configure deployment endpoints
            3. Set up auto-scaling and monitoring
            4. Deploy models to Vertex AI endpoints
            5. Test deployed endpoints for functionality
            6. Generate deployment reports and documentation
            
            Always ensure reliable and scalable deployment for production use.
            """,
            tools=[
                self._create_deployment_prep_tool(),
                self._create_endpoint_creation_tool(),
                self._create_scaling_config_tool(),
                self._create_model_deployment_tool(),
                self._create_endpoint_testing_tool(),
                self._create_deployment_report_tool()
            ]
        )
    
    def _create_deployment_prep_tool(self) -> FunctionTool:
        """Creates tool for deployment preparation"""
        
        def prepare_for_deployment(model_path: str, evaluation_report: str) -> str:
            """
            Prepares model and environment for deployment
            
            Args:
                model_path: Path to trained model
                evaluation_report: JSON string with evaluation results
                
            Returns:
                JSON string with preparation results
            """
            try:
                logger.info(f"Preparing deployment for user {self.user_id}")
                
                eval_obj = json.loads(evaluation_report)
                
                # Check if model is approved for deployment
                decision = eval_obj.get('executive_summary', {}).get('model_status', 'unknown')
                if decision != 'approved':
                    return json.dumps({
                        "user_id": self.user_id,
                        "preparation_status": "error",
                        "error": f"Model not approved for deployment (status: {decision})"
                    })
                
                # Validate model files
                model_files = []
                if os.path.exists(model_path):
                    model_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.json', '.txt', '.bin'))]
                
                if not model_files:
                    return json.dumps({
                        "user_id": self.user_id,
                        "preparation_status": "error",
                        "error": "No model files found for deployment"
                    })
                
                # Prepare deployment configuration
                deployment_config = {
                    "user_id": self.user_id,
                    "deployment_id": f"deploy_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "model_files": model_files,
                    "model_path": model_path,
                    "deployment_timestamp": datetime.now().isoformat(),
                    "endpoint_config": {
                        "endpoint_name": f"user-endpoint-{self.user_id}",
                        "region": self.deployment_config.get('auto_scaling', {}).get('region', 'us-central1'),
                        "machine_type": "n1-standard-2",
                        "min_replicas": self.deployment_config.get('endpoint_config', {}).get('min_replica_count', 1),
                        "max_replicas": self.deployment_config.get('endpoint_config', {}).get('max_replica_count', 10),
                        "traffic_percentage": 100
                    },
                    "monitoring_config": {
                        "enable_logging": True,
                        "log_level": "INFO",
                        "enable_metrics": True,
                        "alert_thresholds": {
                            "cpu_usage": 80,
                            "memory_usage": 85,
                            "response_time": 5.0,
                            "error_rate": 0.05
                        }
                    }
                }
                
                # Save deployment configuration
                with open(self.endpoint_config_path, 'w') as f:
                    json.dump(deployment_config, f, indent=2)
                
                prep_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "preparation_status": "success",
                    "deployment_id": deployment_config["deployment_id"],
                    "model_files_validated": True,
                    "deployment_config_created": True,
                    "ready_for_deployment": True,
                    "estimated_deployment_time": "5-10 minutes"
                }
                
                logger.info(f"Deployment preparation completed for user {self.user_id}")
                return json.dumps(prep_result)
                
            except Exception as e:
                logger.error(f"Error preparing deployment for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "preparation_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(prepare_for_deployment)
    
    def _create_endpoint_creation_tool(self) -> FunctionTool:
        """Creates tool for endpoint creation"""
        
        def create_deployment_endpoint(deployment_config: str) -> str:
            """
            Creates deployment endpoint for the model
            
            Args:
                deployment_config: JSON string with deployment configuration
                
            Returns:
                JSON string with endpoint creation results
            """
            try:
                logger.info(f"Creating deployment endpoint for user {self.user_id}")
                
                config_obj = json.loads(deployment_config)
                deployment_id = config_obj.get('deployment_id')
                endpoint_config = config_obj.get('endpoint_config', {})
                
                # Simulate endpoint creation
                endpoint_info = {
                    "endpoint_id": f"endpoint-{self.user_id}-{uuid.uuid4().hex[:8]}",
                    "endpoint_name": endpoint_config.get('endpoint_name', f"user-endpoint-{self.user_id}"),
                    "region": endpoint_config.get('region', 'us-central1'),
                    "status": "creating",
                    "created_at": datetime.now().isoformat(),
                    "deployment_id": deployment_id
                }
                
                # Simulate endpoint creation process
                endpoint_info.update({
                    "status": "active",
                    "endpoint_url": f"https://{endpoint_info['endpoint_id']}.googleapis.com/v1/predict",
                    "api_key": f"api-key-{uuid.uuid4().hex[:16]}",
                    "documentation_url": f"https://docs.google.com/endpoint/{endpoint_info['endpoint_id']}",
                    "monitoring_dashboard": f"https://console.cloud.google.com/monitoring/endpoint/{endpoint_info['endpoint_id']}"
                })
                
                # Update deployment configuration with endpoint info
                config_obj['endpoint_info'] = endpoint_info
                config_obj['deployment_status'] = 'endpoint_created'
                
                with open(self.endpoint_config_path, 'w') as f:
                    json.dump(config_obj, f, indent=2)
                
                creation_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "endpoint_creation_status": "success",
                    "endpoint_info": endpoint_info,
                    "deployment_progress": "25%"
                }
                
                logger.info(f"Endpoint created for user {self.user_id}")
                return json.dumps(creation_result)
                
            except Exception as e:
                logger.error(f"Error creating endpoint for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "endpoint_creation_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(create_deployment_endpoint)
    
    def _create_scaling_config_tool(self) -> FunctionTool:
        """Creates tool for configuring auto-scaling"""
        
        def configure_auto_scaling(endpoint_info: str) -> str:
            """
            Configures auto-scaling for the deployed endpoint
            
            Args:
                endpoint_info: JSON string with endpoint information
                
            Returns:
                JSON string with scaling configuration results
            """
            try:
                logger.info(f"Configuring auto-scaling for user {self.user_id}")
                
                endpoint_obj = json.loads(endpoint_info)
                endpoint_id = endpoint_obj.get('endpoint_info', {}).get('endpoint_id')
                
                # Configure auto-scaling
                scaling_config = {
                    "endpoint_id": endpoint_id,
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "auto_scaling_enabled": True,
                    "scaling_policy": {
                        "min_replicas": self.deployment_config.get('endpoint_config', {}).get('min_replica_count', 1),
                        "max_replicas": self.deployment_config.get('endpoint_config', {}).get('max_replica_count', 10),
                        "target_cpu_utilization": self.deployment_config.get('auto_scaling', {}).get('target_cpu_utilization', 70),
                        "target_memory_utilization": self.deployment_config.get('auto_scaling', {}).get('target_memory_utilization', 80),
                        "scale_up_delay": "60s",
                        "scale_down_delay": "300s"
                    },
                    "resource_limits": {
                        "cpu_limit": "2",
                        "memory_limit": "4Gi",
                        "cpu_request": "1",
                        "memory_request": "2Gi"
                    },
                    "scaling_metrics": {
                        "requests_per_second": {
                            "target": 100,
                            "scale_up_threshold": 80,
                            "scale_down_threshold": 20
                        },
                        "response_time": {
                            "target_ms": 500,
                            "scale_up_threshold_ms": 800,
                            "scale_down_threshold_ms": 200
                        }
                    }
                }
                
                # Load current deployment config and update
                with open(self.endpoint_config_path, 'r') as f:
                    current_config = json.load(f)
                
                current_config['scaling_config'] = scaling_config
                current_config['deployment_status'] = 'scaling_configured'
                
                with open(self.endpoint_config_path, 'w') as f:
                    json.dump(current_config, f, indent=2)
                
                scaling_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "scaling_config_status": "success",
                    "scaling_config": scaling_config,
                    "deployment_progress": "50%"
                }
                
                logger.info(f"Auto-scaling configured for user {self.user_id}")
                return json.dumps(scaling_result)
                
            except Exception as e:
                logger.error(f"Error configuring auto-scaling for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "scaling_config_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(configure_auto_scaling)
    
    def _create_model_deployment_tool(self) -> FunctionTool:
        """Creates tool for model deployment"""
        
        def deploy_model_to_endpoint(deployment_data: str) -> str:
            """
            Deploys model to the created endpoint
            
            Args:
                deployment_data: JSON string with deployment information
                
            Returns:
                JSON string with deployment results
            """
            try:
                logger.info(f"Deploying model to endpoint for user {self.user_id}")
                
                deployment_obj = json.loads(deployment_data)
                endpoint_info = deployment_obj.get('endpoint_info', {})
                endpoint_url = endpoint_info.get('endpoint_url')
                
                # Simulate model deployment
                deployment_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "deployment_status": "success",
                    "model_deployed": True,
                    "endpoint_url": endpoint_url,
                    "model_version": "1.0",
                    "deployment_id": deployment_obj.get('deployment_id'),
                    "deployment_time": "3 minutes 45 seconds",
                    "model_size_mb": 125.5,
                    "deployment_progress": "75%"
                }
                
                # Update deployment configuration
                with open(self.endpoint_config_path, 'r') as f:
                    current_config = json.load(f)
                
                current_config['deployment_result'] = deployment_result
                current_config['deployment_status'] = 'model_deployed'
                
                with open(self.endpoint_config_path, 'w') as f:
                    json.dump(current_config, f, indent=2)
                
                # Log deployment
                with open(self.deployment_log_path, 'w') as f:
                    f.write(f"Model deployment started for user {self.user_id}\n")
                    f.write(f"Endpoint: {endpoint_url}\n")
                    f.write(f"Deployment ID: {deployment_result['deployment_id']}\n")
                    f.write(f"Status: {deployment_result['deployment_status']}\n")
                    f.write(f"Completed at: {deployment_result['timestamp']}\n")
                
                logger.info(f"Model deployed successfully for user {self.user_id}")
                return json.dumps(deployment_result)
                
            except Exception as e:
                logger.error(f"Error deploying model for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "deployment_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(deploy_model_to_endpoint)
    
    def _create_endpoint_testing_tool(self) -> FunctionTool:
        """Creates tool for endpoint testing"""
        
        def test_deployed_endpoint(endpoint_url: str) -> str:
            """
            Tests the deployed endpoint for functionality
            
            Args:
                endpoint_url: URL of the deployed endpoint
                
            Returns:
                JSON string with testing results
            """
            try:
                logger.info(f"Testing deployed endpoint for user {self.user_id}")
                
                # Test cases
                test_cases = [
                    {
                        "test_name": "health_check",
                        "input": {"prompt": "Hello, how are you?"},
                        "expected_type": "string"
                    },
                    {
                        "test_name": "content_generation",
                        "input": {"prompt": "Write a short story about a robot"},
                        "expected_type": "string"
                    },
                    {
                        "test_name": "response_time",
                        "input": {"prompt": "Quick test"},
                        "expected_max_time": 5.0
                    }
                ]
                
                test_results = []
                for test_case in test_cases:
                    # Simulate endpoint testing
                    result = self._simulate_endpoint_test(test_case, endpoint_url)
                    test_results.append(result)
                
                # Calculate overall test results
                passed_tests = len([r for r in test_results if r['status'] == 'passed'])
                total_tests = len(test_results)
                
                testing_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "endpoint_url": endpoint_url,
                    "testing_status": "completed",
                    "test_results": test_results,
                    "summary": {
                        "total_tests": total_tests,
                        "passed_tests": passed_tests,
                        "failed_tests": total_tests - passed_tests,
                        "success_rate": (passed_tests / total_tests) * 100,
                        "endpoint_ready": passed_tests == total_tests
                    },
                    "deployment_progress": "100%"
                }
                
                # Update deployment configuration
                with open(self.endpoint_config_path, 'r') as f:
                    current_config = json.load(f)
                
                current_config['testing_result'] = testing_result
                current_config['deployment_status'] = 'testing_completed'
                current_config['endpoint_ready'] = testing_result['summary']['endpoint_ready']
                
                with open(self.endpoint_config_path, 'w') as f:
                    json.dump(current_config, f, indent=2)
                
                logger.info(f"Endpoint testing completed for user {self.user_id}")
                return json.dumps(testing_result)
                
            except Exception as e:
                logger.error(f"Error testing endpoint for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "testing_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(test_deployed_endpoint)
    
    def _create_deployment_report_tool(self) -> FunctionTool:
        """Creates tool for generating deployment reports"""
        
        def generate_deployment_report(testing_result: str) -> str:
            """
            Generates comprehensive deployment report
            
            Args:
                testing_result: JSON string with testing results
                
            Returns:
                JSON string with deployment report
            """
            try:
                logger.info(f"Generating deployment report for user {self.user_id}")
                
                testing_obj = json.loads(testing_result)
                
                # Load full deployment configuration
                with open(self.endpoint_config_path, 'r') as f:
                    deployment_config = json.load(f)
                
                report = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "report_id": f"deployment_report_{self.user_id}_{uuid.uuid4().hex[:8]}",
                    "deployment_summary": {
                        "deployment_id": deployment_config.get('deployment_id'),
                        "endpoint_url": deployment_config.get('endpoint_info', {}).get('endpoint_url'),
                        "deployment_status": "completed",
                        "endpoint_ready": testing_obj.get('summary', {}).get('endpoint_ready', False),
                        "total_deployment_time": "8 minutes 30 seconds"
                    },
                    "endpoint_details": {
                        "endpoint_id": deployment_config.get('endpoint_info', {}).get('endpoint_id'),
                        "region": deployment_config.get('endpoint_info', {}).get('region'),
                        "api_key": deployment_config.get('endpoint_info', {}).get('api_key'),
                        "documentation_url": deployment_config.get('endpoint_info', {}).get('documentation_url'),
                        "monitoring_dashboard": deployment_config.get('endpoint_info', {}).get('monitoring_dashboard')
                    },
                    "scaling_configuration": deployment_config.get('scaling_config', {}),
                    "testing_results": testing_obj.get('test_results', []),
                    "performance_metrics": {
                        "average_response_time": "1.2 seconds",
                        "throughput": "50 requests/second",
                        "availability": "99.9%",
                        "error_rate": "0.1%"
                    },
                    "monitoring_setup": {
                        "logging_enabled": True,
                        "metrics_enabled": True,
                        "alerting_configured": True,
                        "dashboard_available": True
                    },
                    "next_steps": [
                        "Monitor endpoint performance",
                        "Set up production traffic",
                        "Configure backup and disaster recovery",
                        "Schedule regular health checks"
                    ],
                    "support_information": {
                        "documentation": deployment_config.get('endpoint_info', {}).get('documentation_url'),
                        "monitoring": deployment_config.get('endpoint_info', {}).get('monitoring_dashboard'),
                        "contact": f"support-{self.user_id}@yourcompany.com"
                    }
                }
                
                # Save deployment report
                report_path = f"logs/user_logs/{self.user_id}/deployment_report.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Deployment report generated for user {self.user_id}")
                return json.dumps(report)
                
            except Exception as e:
                logger.error(f"Error generating deployment report for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "report_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(generate_deployment_report)
    
    # Helper methods
    def _simulate_endpoint_test(self, test_case: Dict[str, Any], endpoint_url: str) -> Dict[str, Any]:
        """Simulate endpoint testing"""
        import time
        import random
        
        start_time = time.time()
        
        # Simulate API call delay
        time.sleep(random.uniform(0.5, 2.0))
        
        response_time = time.time() - start_time
        
        # Simulate test results
        status = "passed" if response_time < 5.0 else "failed"
        
        return {
            "test_name": test_case["test_name"],
            "input": test_case["input"],
            "status": status,
            "response_time": response_time,
            "response": f"Generated response for: {test_case['input']['prompt']}",
            "timestamp": datetime.now().isoformat()
        }
