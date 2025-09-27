import torch
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import requests
from datetime import datetime
import numpy as np

# Configuration
base_model = "gpt2"
lora_model_path = "./cpu_lora_out"
test_data_path = "../UserFacing/db/LLMTestData_CPU.json"
api_base_url = "http://localhost:5000"

class ModelEvaluatorCPU:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.test_data = []
        self.results = {
            "test_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": "CPU",
            "base_model": base_model,
            "lora_path": lora_model_path,
            "test_data_path": test_data_path,
            "device_info": {},
            "test_results": [],
            "summary_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize basic evaluation tools (lightweight for CPU)
        print("✅ CPU Evaluation tools initialized")
    
    def setup_device(self):
        """Setup the appropriate device for inference"""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        self.results["device_info"] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "cpu_count": os.cpu_count()
        }
        
        print(f"✅ Device setup complete: {self.device}")
    
    def load_test_data(self):
        """Load the test dataset"""
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}. Please run training first to generate test data.")
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        print(f"✅ Loaded {len(self.test_data)} test examples")
        self.results["test_data_count"] = len(self.test_data)
    
    def load_model(self):
        """Load the trained LoRA model"""
        if not os.path.exists(lora_model_path):
            raise FileNotFoundError(f"LoRA model not found at {lora_model_path}. Please train the model first using cudaInit_cpu.py")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model).to(self.device)
        
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base, lora_model_path).to(self.device)
        
        print("✅ Model loaded successfully")
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate a response for a given prompt"""
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Disable beam search for speed on CPU
            )
        generation_time = time.time() - start_time
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response_start = full_response.find("### Response:\n")
        if response_start != -1:
            generated_text = full_response[response_start + len("### Response:\n"):].strip()
        else:
            generated_text = full_response.strip()
        
        return generated_text, generation_time
    
    def calculate_basic_metrics(self, generated_text, reference_text):
        """Calculate basic evaluation metrics (CPU-friendly)"""
        metrics = {}
        
        # Basic metrics
        metrics["generated_length"] = len(generated_text)
        metrics["reference_length"] = len(reference_text)
        metrics["length_ratio"] = len(generated_text) / len(reference_text) if len(reference_text) > 0 else 0
        
        # Simple word overlap metrics
        gen_words = set(generated_text.lower().split())
        ref_words = set(reference_text.lower().split())
        
        if len(ref_words) > 0:
            word_overlap = len(gen_words.intersection(ref_words)) / len(ref_words)
            metrics["word_overlap"] = word_overlap
        else:
            metrics["word_overlap"] = 0
        
        # Character-level similarity (simple)
        if len(reference_text) > 0:
            char_similarity = 1 - (abs(len(generated_text) - len(reference_text)) / max(len(generated_text), len(reference_text)))
            metrics["char_similarity"] = char_similarity
        else:
            metrics["char_similarity"] = 0
        
        return metrics
    
    def run_evaluation(self):
        """Run evaluation on all test examples"""
        print(f"\n🧪 Starting evaluation on {len(self.test_data)} test examples...")
        
        total_generation_time = 0
        all_word_overlap_scores = []
        all_char_similarity_scores = []
        all_length_ratios = []
        
        for i, test_example in enumerate(self.test_data):
            prompt = test_example["prompt"]
            # Handle both 'answer' and 'expected_output' keys for compatibility
            reference = test_example.get("answer", test_example.get("expected_output", ""))
            
            print(f"   Testing example {i+1}/{len(self.test_data)}: {prompt[:50]}...")
            
            try:
                generated_text, generation_time = self.generate_response(prompt)
                metrics = self.calculate_basic_metrics(generated_text, reference)
                
                test_result = {
                    "example_id": i + 1,
                    "prompt": prompt,
                    "reference_answer": reference,
                    "generated_answer": generated_text,
                    "generation_time": generation_time,
                    "metrics": metrics
                }
                
                self.results["test_results"].append(test_result)
                
                # Collect metrics for summary
                total_generation_time += generation_time
                if "word_overlap" in metrics:
                    all_word_overlap_scores.append(metrics["word_overlap"])
                if "char_similarity" in metrics:
                    all_char_similarity_scores.append(metrics["char_similarity"])
                if "length_ratio" in metrics:
                    all_length_ratios.append(metrics["length_ratio"])
                
            except Exception as e:
                print(f"❌ Error processing example {i+1}: {e}")
                test_result = {
                    "example_id": i + 1,
                    "prompt": prompt,
                    "reference_answer": reference,
                    "error": str(e)
                }
                self.results["test_results"].append(test_result)
        
        # Calculate summary metrics
        self.results["summary_metrics"] = {
            "total_examples": len(self.test_data),
            "successful_generations": len([r for r in self.results["test_results"] if "generated_answer" in r]),
            "total_generation_time": total_generation_time,
            "average_generation_time": total_generation_time / len(self.test_data) if len(self.test_data) > 0 else 0,
            "average_word_overlap": np.mean(all_word_overlap_scores) if all_word_overlap_scores else None,
            "average_char_similarity": np.mean(all_char_similarity_scores) if all_char_similarity_scores else None,
            "average_length_ratio": np.mean(all_length_ratios) if all_length_ratios else None
        }
        
        print("✅ Evaluation completed!")
    
    def save_results(self):
        """Save evaluation results to file"""
        results_path = f"../UserFacing/db/test_results_cpu_{self.results['test_id']}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Results saved to: {results_path}")
        return results_path
    
    def send_results_to_api(self):
        """Send results to the frontend API"""
        try:
            response = requests.post(
                f"{api_base_url}/test_results",
                json=self.results,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Results sent to frontend API successfully")
                return True
            else:
                print(f"⚠️ API returned status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to send results to API: {e}")
            print("   Make sure the Flask server is running on http://localhost:5000")
            return False
    
    def print_summary(self):
        """Print a summary of the evaluation results"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY (CPU)")
        print("="*60)
        
        metrics = self.results["summary_metrics"]
        
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Successful Generations: {metrics['successful_generations']}")
        print(f"Success Rate: {metrics['successful_generations']/metrics['total_examples']*100:.1f}%")
        print(f"Average Generation Time: {metrics['average_generation_time']:.3f}s")
        
        if metrics['average_word_overlap']:
            print(f"Average Word Overlap: {metrics['average_word_overlap']:.3f}")
        
        if metrics['average_char_similarity']:
            print(f"Average Character Similarity: {metrics['average_char_similarity']:.3f}")
        
        if metrics['average_length_ratio']:
            print(f"Average Length Ratio: {metrics['average_length_ratio']:.3f}")
        
        print("\n" + "="*60)

def main():
    evaluator = ModelEvaluatorCPU()
    
    try:
        # Setup and load everything
        evaluator.setup_device()
        evaluator.load_test_data()
        evaluator.load_model()
        
        # Run evaluation
        evaluator.run_evaluation()
        
        # Save and display results
        results_path = evaluator.save_results()
        evaluator.print_summary()
        
        # Try to send results to API
        evaluator.send_results_to_api()
        
        print(f"\n🎉 Evaluation complete! Results saved to {results_path}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
