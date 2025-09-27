import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import threading
import time
from typing import Optional, Dict, Any

class SLMInference:
    """
    Small Language Model inference class that loads and serves your trained LoRA model
    """
    
    def __init__(self, use_cuda: bool = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.base_model_name = "gpt2"
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Auto-detect device if not specified
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        self.use_cuda = use_cuda
        self._setup_device()
        
    def _setup_device(self):
        """Setup the appropriate device for inference"""
        if self.use_cuda and torch.cuda.is_available():
            self.device = "cuda"
            self.lora_model_path = "../SLMInit/cuda_lora_out"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.lora_model_path = "../SLMInit/cpu_lora_out"
            print("Using Apple Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            self.lora_model_path = "../SLMInit/cpu_lora_out"
            print("Using CPU")
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the trained LoRA model
        Returns status dictionary
        """
        with self.loading_lock:
            if self.model_loaded:
                return {"success": True, "message": "Model already loaded", "device": self.device}
            
            try:
                # Check if LoRA model exists
                if not os.path.exists(self.lora_model_path):
                    return {
                        "success": False, 
                        "error": f"LoRA model not found at {self.lora_model_path}. Please train the model first.",
                        "device": self.device
                    }
                
                print(f"Loading tokenizer for {self.base_model_name}...")
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"Loading base model {self.base_model_name}...")
                # Load base model
                self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
                
                print(f"Loading LoRA adapters from {self.lora_model_path}...")
                # Load LoRA adapters
                self.model = PeftModel.from_pretrained(self.base_model, self.lora_model_path).to(self.device)
                
                self.model_loaded = True
                
                return {
                    "success": True, 
                    "message": f"Model loaded successfully on {self.device}",
                    "device": self.device,
                    "model_path": self.lora_model_path
                }
                
            except Exception as e:
                return {
                    "success": False, 
                    "error": f"Failed to load model: {str(e)}",
                    "device": self.device
                }
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, 
                         do_sample: bool = True) -> Dict[str, Any]:
        """
        Generate a response using the loaded model
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = very random)
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Dictionary with response or error
        """
        if not self.model_loaded:
            load_result = self.load_model()
            if not load_result["success"]:
                return load_result
        
        try:
            # Format prompt in the same style as training
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            print(f"Generating response for: {prompt[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=1 if do_sample else 2,  # Use beam search for non-sampling
                    early_stopping=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after "### Response:\n")
            response_start = full_response.find("### Response:\n")
            if response_start != -1:
                generated_text = full_response[response_start + len("### Response:\n"):].strip()
            else:
                generated_text = full_response.strip()
            
            return {
                "success": True,
                "prompt": prompt,
                "response": generated_text,
                "full_response": full_response,
                "device": self.device,
                "settings": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}",
                "prompt": prompt
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "device": self.device,
            "base_model": self.base_model_name,
            "lora_path": self.lora_model_path if hasattr(self, 'lora_model_path') else None,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        with self.loading_lock:
            if self.model_loaded:
                del self.model
                del self.base_model
                del self.tokenizer
                
                # Clear CUDA cache if using CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self.model = None
                self.base_model = None
                self.tokenizer = None
                self.model_loaded = False
                
                return {"success": True, "message": "Model unloaded successfully"}
            else:
                return {"success": True, "message": "No model was loaded"}

# Global instance
slm_inference = SLMInference()

def generate_slm_stream(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
    """
    Generator function for streaming SLM responses
    """
    try:
        # Load model if not loaded
        if not slm_inference.model_loaded:
            yield f"data: {json.dumps({'status': 'Loading SLM model...'})}\n\n"
            load_result = slm_inference.load_model()
            if not load_result["success"]:
                yield f"data: {json.dumps({'error': load_result['error']})}\n\n"
                return
            yield f"data: {json.dumps({'status': f'Model loaded on {load_result[\"device\"]}'})}\n\n"
        
        yield f"data: {json.dumps({'status': 'Generating response...'})}\n\n"
        
        # Generate response
        result = slm_inference.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        if result["success"]:
            # Stream the response word by word for better UX
            response_text = result["response"]
            words = response_text.split()
            full_text = ""
            
            for i, word in enumerate(words):
                full_text += word + " "
                yield f"data: {json.dumps({'text': word + ' ', 'full_text': full_text.strip(), 'is_complete': i == len(words) - 1})}\n\n"
                time.sleep(0.05)  # Small delay for streaming effect
            
            # Save to database files (same format as existing system)
            try:
                json_data = {
                    "prompt": prompt,
                    "answer": response_text
                }
                
                # Write to LLMCurrData.json (replace with most recent)
                with open("db/LLMCurrData.json", "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                # Append to LLMData.json (accumulate all messages)
                existing_data = []
                try:
                    with open("db/LLMData.json", "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []
                
                if not isinstance(existing_data, list):
                    existing_data = []
                
                existing_data.append(json_data)
                
                with open("db/LLMData.json", "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                yield f"data: {json.dumps({'status': 'Response saved to database!'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': f'Warning: Could not save to database: {str(e)}'})}\n\n"
            
            yield f"data: {json.dumps({'status': 'Response complete!', 'device_used': result['device']})}\n\n"
        else:
            yield f"data: {json.dumps({'error': result['error']})}\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Streaming failed: {str(e)}'})}\n\n"
