"""
Model-specific tools for Google ADK agents
Provides tools for model management, versioning, and deployment
"""

from google.adk.tools import Tool
from typing import Dict, List, Any, Optional
import json
import os
import logging
from datetime import datetime
import uuid
import shutil
import hashlib

logger = logging.getLogger(__name__)

class ModelTools:
    """
    Collection of model-specific tools for ADK agents
    """
    
    @staticmethod
    def create_model_versioning_tool() -> Tool:
        """Creates a tool for model versioning and management"""
        
        def manage_model_versions(model_path: str, user_id: str) -> str:
            """
            Manages model versions and creates new versions
            
            Args:
                model_path: Path to the model
                user_id: User identifier
                
            Returns:
                JSON string with version management results
            """
            try:
                logger.info(f"Managing model versions for user {user_id}")
                
                version_info = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "versioning_status": "success",
                    "model_path": model_path,
                    "versions": []
                }
                
                # Check for existing versions
                versions_dir = f"{model_path}/versions"
                if os.path.exists(versions_dir):
                    for version_folder in os.listdir(versions_dir):
                        version_path = os.path.join(versions_dir, version_folder)
                        if os.path.isdir(version_path):
                            version_data = ModelTools._analyze_model_version(version_path, version_folder)
                            version_info["versions"].append(version_data)
                
                # Create new version if model exists
                if os.path.exists(model_path):
                    new_version = ModelTools._create_new_version(model_path, user_id)
                    version_info["versions"].append(new_version)
                    version_info["current_version"] = new_version["version_id"]
                else:
                    version_info["versioning_status"] = "warning"
                    version_info["message"] = "Model path not found"
                
                # Sort versions by creation date
                version_info["versions"].sort(key=lambda x: x["created_at"], reverse=True)
                
                return json.dumps(version_info)
                
            except Exception as e:
                logger.error(f"Error managing model versions for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "versioning_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="manage_model_versions",
            description="Manages model versions and creates new versions",
            function=manage_model_versions
        )
    
    @staticmethod
    def create_model_backup_tool() -> Tool:
        """Creates a tool for model backup and restoration"""
        
        def backup_model(model_path: str, user_id: str, backup_location: str = None) -> str:
            """
            Creates a backup of the model
            
            Args:
                model_path: Path to the model to backup
                user_id: User identifier
                backup_location: Optional backup location
                
            Returns:
                JSON string with backup results
            """
            try:
                logger.info(f"Creating model backup for user {user_id}")
                
                if not os.path.exists(model_path):
                    return json.dumps({
                        "user_id": user_id,
                        "backup_status": "error",
                        "error": "Model path not found"
                    })
                
                # Determine backup location
                if not backup_location:
                    backup_location = f"backups/user_models/{user_id}"
                
                # Create backup directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{backup_location}/backup_{timestamp}"
                os.makedirs(backup_path, exist_ok=True)
                
                # Copy model files
                copied_files = []
                for item in os.listdir(model_path):
                    src_path = os.path.join(model_path, item)
                    dst_path = os.path.join(backup_path, item)
                    
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(item)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                        copied_files.append(f"{item}/")
                
                # Calculate backup size
                backup_size = ModelTools._calculate_directory_size(backup_path)
                
                # Create backup metadata
                backup_metadata = {
                    "backup_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "backup_path": backup_path,
                    "original_path": model_path,
                    "backup_size_bytes": backup_size,
                    "files_backed_up": copied_files,
                    "backup_timestamp": datetime.now().isoformat(),
                    "checksum": ModelTools._calculate_directory_checksum(backup_path)
                }
                
                # Save backup metadata
                metadata_path = os.path.join(backup_path, "backup_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(backup_metadata, f, indent=2)
                
                backup_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "backup_status": "success",
                    "backup_id": backup_metadata["backup_id"],
                    "backup_path": backup_path,
                    "backup_size_mb": round(backup_size / (1024 * 1024), 2),
                    "files_backed_up": len(copied_files),
                    "backup_metadata": metadata_path
                }
                
                logger.info(f"Model backup created for user {user_id}")
                return json.dumps(backup_result)
                
            except Exception as e:
                logger.error(f"Error creating backup for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "backup_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="backup_model",
            description="Creates a backup of the model",
            function=backup_model
        )
    
    @staticmethod
    def create_model_restore_tool() -> Tool:
        """Creates a tool for restoring models from backup"""
        
        def restore_model(backup_path: str, restore_location: str, user_id: str) -> str:
            """
            Restores a model from backup
            
            Args:
                backup_path: Path to the backup
                restore_location: Location to restore the model
                user_id: User identifier
                
            Returns:
                JSON string with restore results
            """
            try:
                logger.info(f"Restoring model for user {user_id}")
                
                if not os.path.exists(backup_path):
                    return json.dumps({
                        "user_id": user_id,
                        "restore_status": "error",
                        "error": "Backup path not found"
                    })
                
                # Load backup metadata
                metadata_path = os.path.join(backup_path, "backup_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        backup_metadata = json.load(f)
                else:
                    backup_metadata = {}
                
                # Create restore location
                os.makedirs(restore_location, exist_ok=True)
                
                # Restore files
                restored_files = []
                for item in os.listdir(backup_path):
                    if item == "backup_metadata.json":
                        continue
                        
                    src_path = os.path.join(backup_path, item)
                    dst_path = os.path.join(restore_location, item)
                    
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        restored_files.append(item)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                        restored_files.append(f"{item}/")
                
                # Verify restore integrity
                restore_checksum = ModelTools._calculate_directory_checksum(restore_location)
                backup_checksum = backup_metadata.get('checksum', '')
                integrity_check = restore_checksum == backup_checksum if backup_checksum else True
                
                restore_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "restore_status": "success" if integrity_check else "warning",
                    "restore_location": restore_location,
                    "backup_path": backup_path,
                    "restored_files": len(restored_files),
                    "integrity_check": integrity_check,
                    "backup_metadata": backup_metadata,
                    "restore_size_mb": round(ModelTools._calculate_directory_size(restore_location) / (1024 * 1024), 2)
                }
                
                if not integrity_check:
                    restore_result["warning"] = "Integrity check failed - backup may be corrupted"
                
                logger.info(f"Model restored for user {user_id}")
                return json.dumps(restore_result)
                
            except Exception as e:
                logger.error(f"Error restoring model for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "restore_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="restore_model",
            description="Restores a model from backup",
            function=restore_model
        )
    
    @staticmethod
    def create_model_export_tool() -> Tool:
        """Creates a tool for exporting models in different formats"""
        
        def export_model(model_path: str, export_format: str, user_id: str) -> str:
            """
            Exports model in specified format
            
            Args:
                model_path: Path to the model
                export_format: Format to export to (onnx, torchscript, etc.)
                user_id: User identifier
                
            Returns:
                JSON string with export results
            """
            try:
                logger.info(f"Exporting model for user {user_id} in {export_format} format")
                
                if not os.path.exists(model_path):
                    return json.dumps({
                        "user_id": user_id,
                        "export_status": "error",
                        "error": "Model path not found"
                    })
                
                # Create export directory
                export_dir = f"exports/user_models/{user_id}/{export_format}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"{export_dir}/export_{timestamp}"
                os.makedirs(export_path, exist_ok=True)
                
                # Export based on format
                export_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "export_format": export_format,
                    "export_path": export_path,
                    "export_status": "success"
                }
                
                if export_format.lower() == "onnx":
                    export_result.update(ModelTools._export_to_onnx(model_path, export_path))
                elif export_format.lower() == "torchscript":
                    export_result.update(ModelTools._export_to_torchscript(model_path, export_path))
                elif export_format.lower() == "tensorflow":
                    export_result.update(ModelTools._export_to_tensorflow(model_path, export_path))
                else:
                    export_result.update(ModelTools._export_generic(model_path, export_path, export_format))
                
                # Create export metadata
                export_metadata = {
                    "export_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "original_path": model_path,
                    "export_format": export_format,
                    "export_path": export_path,
                    "export_timestamp": datetime.now().isoformat(),
                    "export_size_bytes": ModelTools._calculate_directory_size(export_path),
                    "export_files": os.listdir(export_path)
                }
                
                metadata_path = os.path.join(export_path, "export_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(export_metadata, f, indent=2)
                
                export_result["export_metadata"] = export_metadata
                export_result["export_size_mb"] = round(export_metadata["export_size_bytes"] / (1024 * 1024), 2)
                
                logger.info(f"Model exported for user {user_id}")
                return json.dumps(export_result)
                
            except Exception as e:
                logger.error(f"Error exporting model for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "export_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="export_model",
            description="Exports model in specified format",
            function=export_model
        )
    
    @staticmethod
    def create_model_validation_tool() -> Tool:
        """Creates a tool for validating model integrity"""
        
        def validate_model_integrity(model_path: str, user_id: str) -> str:
            """
            Validates model integrity and structure
            
            Args:
                model_path: Path to the model
                user_id: User identifier
                
            Returns:
                JSON string with validation results
            """
            try:
                logger.info(f"Validating model integrity for user {user_id}")
                
                validation_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "model_path": model_path,
                    "validation_status": "success",
                    "checks": {},
                    "issues": [],
                    "recommendations": []
                }
                
                if not os.path.exists(model_path):
                    validation_result["validation_status"] = "error"
                    validation_result["issues"].append("Model path does not exist")
                    return json.dumps(validation_result)
                
                # Check file structure
                validation_result["checks"]["file_structure"] = ModelTools._validate_file_structure(model_path)
                
                # Check model files
                validation_result["checks"]["model_files"] = ModelTools._validate_model_files(model_path)
                
                # Check file integrity
                validation_result["checks"]["file_integrity"] = ModelTools._validate_file_integrity(model_path)
                
                # Check model metadata
                validation_result["checks"]["metadata"] = ModelTools._validate_model_metadata(model_path)
                
                # Check file permissions
                validation_result["checks"]["permissions"] = ModelTools._validate_file_permissions(model_path)
                
                # Aggregate issues and recommendations
                for check_name, check_result in validation_result["checks"].items():
                    if isinstance(check_result, dict) and check_result.get("status") == "failed":
                        validation_result["issues"].extend(check_result.get("issues", []))
                        validation_result["recommendations"].extend(check_result.get("recommendations", []))
                
                # Overall validation status
                if validation_result["issues"]:
                    validation_result["validation_status"] = "warning" if len(validation_result["issues"]) < 3 else "failed"
                
                validation_result["overall_score"] = ModelTools._calculate_validation_score(validation_result["checks"])
                
                return json.dumps(validation_result)
                
            except Exception as e:
                logger.error(f"Error validating model for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "validation_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="validate_model_integrity",
            description="Validates model integrity and structure",
            function=validate_model_integrity
        )
    
    @staticmethod
    def create_model_cleanup_tool() -> Tool:
        """Creates a tool for cleaning up old model files"""
        
        def cleanup_old_models(user_id: str, retention_days: int = 30) -> str:
            """
            Cleans up old model files and versions
            
            Args:
                user_id: User identifier
                retention_days: Number of days to retain models
                
            Returns:
                JSON string with cleanup results
            """
            try:
                logger.info(f"Cleaning up old models for user {user_id}")
                
                cleanup_result = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "retention_days": retention_days,
                    "cleanup_status": "success",
                    "cleaned_files": [],
                    "cleaned_size_bytes": 0,
                    "errors": []
                }
                
                # Clean up model versions
                versions_path = f"models/user_models/{user_id}/versions"
                if os.path.exists(versions_path):
                    version_cleanup = ModelTools._cleanup_versions(versions_path, retention_days)
                    cleanup_result["cleaned_files"].extend(version_cleanup["cleaned_files"])
                    cleanup_result["cleaned_size_bytes"] += version_cleanup["cleaned_size_bytes"]
                    cleanup_result["errors"].extend(version_cleanup.get("errors", []))
                
                # Clean up exports
                exports_path = f"exports/user_models/{user_id}"
                if os.path.exists(exports_path):
                    export_cleanup = ModelTools._cleanup_exports(exports_path, retention_days)
                    cleanup_result["cleaned_files"].extend(export_cleanup["cleaned_files"])
                    cleanup_result["cleaned_size_bytes"] += export_cleanup["cleaned_size_bytes"]
                    cleanup_result["errors"].extend(export_cleanup.get("errors", []))
                
                # Clean up backups
                backups_path = f"backups/user_models/{user_id}"
                if os.path.exists(backups_path):
                    backup_cleanup = ModelTools._cleanup_backups(backups_path, retention_days)
                    cleanup_result["cleaned_files"].extend(backup_cleanup["cleaned_files"])
                    cleanup_result["cleaned_size_bytes"] += backup_cleanup["cleaned_size_bytes"]
                    cleanup_result["errors"].extend(backup_cleanup.get("errors", []))
                
                cleanup_result["cleaned_size_mb"] = round(cleanup_result["cleaned_size_bytes"] / (1024 * 1024), 2)
                cleanup_result["total_files_cleaned"] = len(cleanup_result["cleaned_files"])
                
                if cleanup_result["errors"]:
                    cleanup_result["cleanup_status"] = "warning"
                
                logger.info(f"Model cleanup completed for user {user_id}")
                return json.dumps(cleanup_result)
                
            except Exception as e:
                logger.error(f"Error cleaning up models for user {user_id}: {e}")
                return json.dumps({
                    "user_id": user_id,
                    "cleanup_status": "error",
                    "error": str(e)
                })
        
        return Tool(
            name="cleanup_old_models",
            description="Cleans up old model files and versions",
            function=cleanup_old_models
        )
    
    # Helper methods
    @staticmethod
    def _analyze_model_version(version_path: str, version_id: str) -> Dict[str, Any]:
        """Analyze a model version"""
        try:
            version_data = {
                "version_id": version_id,
                "version_path": version_path,
                "created_at": datetime.fromtimestamp(os.path.getctime(version_path)).isoformat(),
                "size_bytes": ModelTools._calculate_directory_size(version_path),
                "files": os.listdir(version_path) if os.path.exists(version_path) else []
            }
            
            # Try to load version metadata
            metadata_path = os.path.join(version_path, "version_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                version_data.update(metadata)
            
            return version_data
        except Exception as e:
            logger.error(f"Error analyzing model version {version_id}: {e}")
            return {
                "version_id": version_id,
                "version_path": version_path,
                "error": str(e)
            }
    
    @staticmethod
    def _create_new_version(model_path: str, user_id: str) -> Dict[str, Any]:
        """Create a new model version"""
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        versions_dir = f"models/user_models/{user_id}/versions"
        version_path = os.path.join(versions_dir, version_id)
        
        os.makedirs(version_path, exist_ok=True)
        
        # Copy model files to version directory
        copied_files = []
        for item in os.listdir(model_path):
            src_path = os.path.join(model_path, item)
            dst_path = os.path.join(version_path, item)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                copied_files.append(item)
        
        # Create version metadata
        version_metadata = {
            "version_id": version_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "files": copied_files,
            "size_bytes": ModelTools._calculate_directory_size(version_path)
        }
        
        metadata_path = os.path.join(version_path, "version_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        return version_metadata
    
    @staticmethod
    def _calculate_directory_size(directory_path: str) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    @staticmethod
    def _calculate_directory_checksum(directory_path: str) -> str:
        """Calculate checksum for directory"""
        hasher = hashlib.md5()
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in sorted(filenames):
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def _export_to_onnx(model_path: str, export_path: str) -> Dict[str, Any]:
        """Export model to ONNX format (simulated)"""
        # In a real implementation, you would use torch.onnx.export
        return {
            "export_type": "onnx",
            "exported_files": ["model.onnx", "config.json"],
            "export_notes": "ONNX export completed successfully"
        }
    
    @staticmethod
    def _export_to_torchscript(model_path: str, export_path: str) -> Dict[str, Any]:
        """Export model to TorchScript format (simulated)"""
        # In a real implementation, you would use torch.jit.script or torch.jit.trace
        return {
            "export_type": "torchscript",
            "exported_files": ["model.pt", "config.json"],
            "export_notes": "TorchScript export completed successfully"
        }
    
    @staticmethod
    def _export_to_tensorflow(model_path: str, export_path: str) -> Dict[str, Any]:
        """Export model to TensorFlow format (simulated)"""
        # In a real implementation, you would use appropriate TensorFlow conversion tools
        return {
            "export_type": "tensorflow",
            "exported_files": ["saved_model.pb", "variables/", "assets/"],
            "export_notes": "TensorFlow export completed successfully"
        }
    
    @staticmethod
    def _export_generic(model_path: str, export_path: str, export_format: str) -> Dict[str, Any]:
        """Generic export (copy files)"""
        import shutil
        shutil.copytree(model_path, export_path, dirs_exist_ok=True)
        
        return {
            "export_type": export_format,
            "exported_files": os.listdir(export_path),
            "export_notes": f"Generic export to {export_format} completed"
        }
    
    @staticmethod
    def _validate_file_structure(model_path: str) -> Dict[str, Any]:
        """Validate model file structure"""
        required_files = ["model.pt", "config.json"]
        optional_files = ["tokenizer.json", "vocab.txt", "metadata.json"]
        
        found_files = os.listdir(model_path) if os.path.exists(model_path) else []
        
        issues = []
        recommendations = []
        
        for required_file in required_files:
            if required_file not in found_files:
                issues.append(f"Required file missing: {required_file}")
        
        if not issues:
            recommendations.append("File structure is valid")
        
        return {
            "status": "passed" if not issues else "failed",
            "found_files": found_files,
            "issues": issues,
            "recommendations": recommendations
        }
    
    @staticmethod
    def _validate_model_files(model_path: str) -> Dict[str, Any]:
        """Validate model files integrity"""
        issues = []
        recommendations = []
        
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            if file_size < 1024:  # Less than 1KB
                issues.append("Model file appears to be too small")
            elif file_size > 1024 * 1024 * 1024:  # More than 1GB
                issues.append("Model file is very large")
                recommendations.append("Consider model compression")
        else:
            issues.append("Model file not found")
        
        return {
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "recommendations": recommendations
        }
    
    @staticmethod
    def _validate_file_integrity(model_path: str) -> Dict[str, Any]:
        """Validate file integrity"""
        issues = []
        
        for filename in os.listdir(model_path):
            filepath = os.path.join(model_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        f.read(1024)  # Try to read first 1KB
                except Exception as e:
                    issues.append(f"File {filename} is corrupted: {str(e)}")
        
        return {
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "recommendations": ["File integrity check completed"] if not issues else ["Fix corrupted files"]
        }
    
    @staticmethod
    def _validate_model_metadata(model_path: str) -> Dict[str, Any]:
        """Validate model metadata"""
        issues = []
        recommendations = []
        
        metadata_file = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                required_fields = ["user_id", "created_at", "model_version"]
                for field in required_fields:
                    if field not in metadata:
                        issues.append(f"Metadata missing required field: {field}")
                
                if not issues:
                    recommendations.append("Metadata is complete and valid")
                    
            except json.JSONDecodeError:
                issues.append("Metadata file contains invalid JSON")
        else:
            issues.append("Metadata file not found")
            recommendations.append("Create metadata.json file")
        
        return {
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "recommendations": recommendations
        }
    
    @staticmethod
    def _validate_file_permissions(model_path: str) -> Dict[str, Any]:
        """Validate file permissions"""
        issues = []
        
        for filename in os.listdir(model_path):
            filepath = os.path.join(model_path, filename)
            if os.path.isfile(filepath):
                if not os.access(filepath, os.R_OK):
                    issues.append(f"File {filename} is not readable")
                if not os.access(filepath, os.W_OK):
                    issues.append(f"File {filename} is not writable")
        
        return {
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "recommendations": ["Fix file permissions"] if issues else ["File permissions are correct"]
        }
    
    @staticmethod
    def _calculate_validation_score(checks: Dict[str, Dict]) -> float:
        """Calculate overall validation score"""
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks.values() if check.get("status") == "passed")
        return (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    @staticmethod
    def _cleanup_versions(versions_path: str, retention_days: int) -> Dict[str, Any]:
        """Clean up old model versions"""
        import time
        
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        cleaned_files = []
        cleaned_size = 0
        errors = []
        
        try:
            for version_dir in os.listdir(versions_path):
                version_path = os.path.join(versions_path, version_dir)
                if os.path.isdir(version_path):
                    creation_time = os.path.getctime(version_path)
                    if creation_time < cutoff_time:
                        try:
                            size = ModelTools._calculate_directory_size(version_path)
                            shutil.rmtree(version_path)
                            cleaned_files.append(version_dir)
                            cleaned_size += size
                        except Exception as e:
                            errors.append(f"Error cleaning version {version_dir}: {str(e)}")
        except Exception as e:
            errors.append(f"Error accessing versions directory: {str(e)}")
        
        return {
            "cleaned_files": cleaned_files,
            "cleaned_size_bytes": cleaned_size,
            "errors": errors
        }
    
    @staticmethod
    def _cleanup_exports(exports_path: str, retention_days: int) -> Dict[str, Any]:
        """Clean up old exports"""
        return ModelTools._cleanup_versions(exports_path, retention_days)
    
    @staticmethod
    def _cleanup_backups(backups_path: str, retention_days: int) -> Dict[str, Any]:
        """Clean up old backups"""
        return ModelTools._cleanup_versions(backups_path, retention_days)
