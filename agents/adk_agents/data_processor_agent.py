"""
Data Processor Agent using Google ADK
Processes and prepares training data for user model fine-tuning
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
import json
import os
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessorAgent:
    """
    Agent responsible for processing and preparing user training data
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.processed_data_path = f"data/user_data/{user_id}/processed_data.json"
        
    def get_agent(self) -> Agent:
        """
        Creates and returns the data processor agent
        """
        return Agent(
            name=f"data_processor_{self.user_id.replace('-', '_')}",
            description="Processes and prepares training data for user model fine-tuning",
            model="gemini-2.0-flash",
            instruction=f"""
            You are a data processing agent for user {self.user_id}.
            Your responsibilities include:
            1. Analyze and validate the provided training data
            2. Clean and format the data appropriately
            3. Prepare data for fine-tuning (tokenization, formatting)
            4. Save processed data to the appropriate location
            5. Report data quality metrics and statistics
            
            Always ensure data quality and proper formatting for machine learning training.
            """,
            tools=[
                self._create_data_validation_tool(),
                self._create_data_cleaning_tool(),
                self._create_data_formatting_tool(),
                self._create_data_saving_tool(),
                self._create_data_metrics_tool()
            ]
        )
    
    def _create_data_validation_tool(self) -> FunctionTool:
        """Creates tool for data validation"""
        
        def validate_training_data(raw_data: str) -> str:
            """
            Validates the quality and format of training data
            
            Args:
                raw_data: Raw training data string
                
            Returns:
                JSON string with validation results
            """
            try:
                logger.info(f"Validating training data for user {self.user_id}")
                
                validation_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "validation_status": "success",
                    "data_size": len(raw_data),
                    "line_count": len(raw_data.split('\n')),
                    "word_count": len(raw_data.split()),
                    "character_count": len(raw_data),
                    "issues": [],
                    "recommendations": []
                }
                
                # Check for minimum data size
                if len(raw_data) < 100:
                    validation_result["issues"].append("Training data too small (minimum 100 characters recommended)")
                    validation_result["validation_status"] = "warning"
                
                # Check for empty lines
                empty_lines = raw_data.count('\n\n')
                if empty_lines > 0:
                    validation_result["issues"].append(f"Found {empty_lines} empty lines")
                
                # Check for special characters
                special_chars = re.findall(r'[^\w\s.,!?;:\'"-]', raw_data)
                if special_chars:
                    unique_special = list(set(special_chars))
                    validation_result["issues"].append(f"Found special characters: {unique_special}")
                
                # Recommendations
                if validation_result["word_count"] < 50:
                    validation_result["recommendations"].append("Consider adding more training data for better model performance")
                
                if validation_result["line_count"] < 5:
                    validation_result["recommendations"].append("Consider breaking data into more examples for better training")
                
                logger.info(f"Data validation completed for user {self.user_id}")
                return json.dumps(validation_result)
                
            except Exception as e:
                logger.error(f"Error validating data for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "validation_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(validate_training_data)
    
    def _create_data_cleaning_tool(self) -> FunctionTool:
        """Creates tool for data cleaning"""
        
        def clean_training_data(raw_data: str) -> str:
            """
            Cleans and normalizes training data
            
            Args:
                raw_data: Raw training data string
                
            Returns:
                JSON string with cleaned data and cleaning report
            """
            try:
                logger.info(f"Cleaning training data for user {self.user_id}")
                
                # Clean the data
                cleaned_data = raw_data.strip()
                
                # Remove excessive whitespace
                cleaned_data = re.sub(r'\s+', ' ', cleaned_data)
                
                # Remove excessive newlines
                cleaned_data = re.sub(r'\n\s*\n', '\n', cleaned_data)
                
                # Normalize quotes
                cleaned_data = cleaned_data.replace('"', '"').replace('"', '"')
                cleaned_data = cleaned_data.replace(''', "'").replace(''', "'")
                
                # Remove control characters
                cleaned_data = ''.join(char for char in cleaned_data if ord(char) >= 32 or char in '\n\t')
                
                cleaning_report = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "original_size": len(raw_data),
                    "cleaned_size": len(cleaned_data),
                    "bytes_removed": len(raw_data) - len(cleaned_data),
                    "cleaning_operations": [
                        "Whitespace normalization",
                        "Newline normalization", 
                        "Quote normalization",
                        "Control character removal"
                    ],
                    "cleaned_data": cleaned_data
                }
                
                logger.info(f"Data cleaning completed for user {self.user_id}")
                return json.dumps(cleaning_report)
                
            except Exception as e:
                logger.error(f"Error cleaning data for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "cleaning_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(clean_training_data)
    
    def _create_data_formatting_tool(self) -> FunctionTool:
        """Creates tool for data formatting"""
        
        def format_training_data(cleaned_data: str) -> str:
            """
            Formats data for fine-tuning (creates training examples)
            
            Args:
                cleaned_data: Cleaned training data string
                
            Returns:
                JSON string with formatted training examples
            """
            try:
                logger.info(f"Formatting training data for user {self.user_id}")
                
                # Split data into sentences/examples
                sentences = re.split(r'[.!?]\s+', cleaned_data)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # Create training examples
                training_examples = []
                for i, sentence in enumerate(sentences):
                    if len(sentence) > 10:  # Only include meaningful sentences
                        example = {
                            "id": f"{self.user_id}_example_{i+1}",
                            "text": sentence,
                            "length": len(sentence),
                            "word_count": len(sentence.split())
                        }
                        training_examples.append(example)
                
                # Format for different model types
                formatted_data = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_examples": len(training_examples),
                    "examples": training_examples,
                    "formats": {
                        "text_only": [ex["text"] for ex in training_examples],
                        "jsonl": "\n".join([json.dumps(ex) for ex in training_examples]),
                        "prompt_completion": [
                            {"prompt": ex["text"][:len(ex["text"])//2], "completion": ex["text"][len(ex["text"])//2:]}
                            for ex in training_examples if len(ex["text"]) > 20
                        ]
                    }
                }
                
                logger.info(f"Data formatting completed for user {self.user_id}: {len(training_examples)} examples")
                return json.dumps(formatted_data)
                
            except Exception as e:
                logger.error(f"Error formatting data for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "formatting_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(format_training_data)
    
    def _create_data_saving_tool(self) -> FunctionTool:
        """Creates tool for saving processed data"""
        
        def save_processed_data(formatted_data: str) -> str:
            """
            Saves processed data to storage
            
            Args:
                formatted_data: JSON string with formatted data
                
            Returns:
                JSON string with save operation results
            """
            try:
                logger.info(f"Saving processed data for user {self.user_id}")
                
                # Ensure directory exists
                os.makedirs(f"data/user_data/{self.user_id}", exist_ok=True)
                
                # Save formatted data
                with open(self.processed_data_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_data)
                
                # Save raw text version
                data_obj = json.loads(formatted_data)
                raw_text_path = f"data/user_data/{self.user_id}/raw_text.txt"
                with open(raw_text_path, 'w', encoding='utf-8') as f:
                    for example in data_obj.get('examples', []):
                        f.write(example['text'] + '\n')
                
                save_result = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "save_status": "success",
                    "files_saved": [
                        self.processed_data_path,
                        raw_text_path
                    ],
                    "total_examples": data_obj.get('total_examples', 0),
                    "file_sizes": {
                        "processed_json": os.path.getsize(self.processed_data_path),
                        "raw_text": os.path.getsize(raw_text_path)
                    }
                }
                
                logger.info(f"Data saved successfully for user {self.user_id}")
                return json.dumps(save_result)
                
            except Exception as e:
                logger.error(f"Error saving data for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "save_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(save_processed_data)
    
    def _create_data_metrics_tool(self) -> FunctionTool:
        """Creates tool for data metrics and statistics"""
        
        def calculate_data_metrics(data_path: str) -> str:
            """
            Calculates metrics and statistics for processed data
            
            Args:
                data_path: Path to processed data file
                
            Returns:
                JSON string with data metrics
            """
            try:
                logger.info(f"Calculating data metrics for user {self.user_id}")
                
                if not os.path.exists(data_path):
                    return json.dumps({
                        "user_id": self.user_id,
                        "metrics_status": "error",
                        "error": "Data file not found"
                    })
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    data_obj = json.load(f)
                
                examples = data_obj.get('examples', [])
                
                # Calculate metrics
                total_chars = sum(ex['length'] for ex in examples)
                total_words = sum(ex['word_count'] for ex in examples)
                avg_sentence_length = total_chars / len(examples) if examples else 0
                avg_word_count = total_words / len(examples) if examples else 0
                
                metrics = {
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_examples": len(examples),
                    "total_characters": total_chars,
                    "total_words": total_words,
                    "average_sentence_length": round(avg_sentence_length, 2),
                    "average_word_count": round(avg_word_count, 2),
                    "data_quality_score": min(100, max(0, len(examples) * 10)),  # Simple quality score
                    "recommendations": []
                }
                
                # Add recommendations based on metrics
                if len(examples) < 10:
                    metrics["recommendations"].append("Consider adding more training examples")
                if avg_sentence_length < 20:
                    metrics["recommendations"].append("Consider longer training examples")
                if avg_sentence_length > 200:
                    metrics["recommendations"].append("Consider shorter training examples")
                
                logger.info(f"Data metrics calculated for user {self.user_id}")
                return json.dumps(metrics)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for user {self.user_id}: {e}")
                return json.dumps({
                    "user_id": self.user_id,
                    "metrics_status": "error",
                    "error": str(e)
                })
        
        return FunctionTool(calculate_data_metrics)
