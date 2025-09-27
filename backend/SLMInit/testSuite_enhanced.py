import torch
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import requests
from datetime import datetime
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
base_model = "gpt2"
lora_model_path = "./cuda_lora_out"
test_data_path = "../UserFacing/db/LLMTestData.json"
api_base_url = "http://localhost:5000"

class ModelEvaluator:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.test_data = []
        self.results = {
            "test_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": "CUDA",
            "base_model": base_model,
            "lora_path": lora_model_path,
            "test_data_path": test_data_path,
            "device_info": {},
            "test_results": [],
            "summary_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize evaluation tools
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Evaluation tools initialized")
        except Exception as e:
            print(f"⚠️ Warning: Some evaluation tools not available: {e}")
            self.rouge_scorer = None
            self.sentence_model = None
    
    def setup_device(self):
        """Setup the appropriate device for inference"""
        if not torch.cuda.is_available():
            print("CUDA is not available. This could be due to:")
            print("1. PyTorch was installed without CUDA support")
            print("2. CUDA drivers are not installed")
            print("3. CUDA version mismatch")
            print("\nTo fix this:")
            print("1. Uninstall current PyTorch: pip uninstall torch")
            print("2. Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("3. Or use the requirements.txt: pip install -r requirements.txt")
            raise RuntimeError("CUDA is not available. This script requires CUDA for inference.")
        
        self.device = "cuda"
        self.results["device_info"] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        print(f"✅ Device setup complete: {self.device}")
        print(f"   CUDA Device: {torch.cuda.get_device_name()}")
    
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
            raise FileNotFoundError(f"LoRA model not found at {lora_model_path}. Please train the model first using cudaInit.py")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model).to(self.device)
        
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base, lora_model_path).to(self.device)
        
        print("✅ Model loaded successfully")
    
    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7):
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
                num_beams=1,
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
    
    def calculate_metrics(self, generated_text, reference_text):
        """Calculate various evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics["generated_length"] = len(generated_text)
        metrics["reference_length"] = len(reference_text)
        metrics["length_ratio"] = len(generated_text) / len(reference_text) if len(reference_text) > 0 else 0
        
        # ROUGE scores (if available)
        if self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(reference_text, generated_text)
                metrics["rouge1_f"] = rouge_scores['rouge1'].fmeasure
                metrics["rouge2_f"] = rouge_scores['rouge2'].fmeasure
                metrics["rougeL_f"] = rouge_scores['rougeL'].fmeasure
            except Exception as e:
                print(f"⚠️ ROUGE calculation failed: {e}")
        
        # Semantic similarity (if available)
        if self.sentence_model:
            try:
                ref_embedding = self.sentence_model.encode([reference_text])
                gen_embedding = self.sentence_model.encode([generated_text])
                similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
                metrics["semantic_similarity"] = float(similarity)
            except Exception as e:
                print(f"⚠️ Semantic similarity calculation failed: {e}")
        
        return metrics
    
    def run_evaluation(self):
        """Run evaluation on all test examples"""
        print(f"\n🧪 Starting evaluation on {len(self.test_data)} test examples...")
        
        total_generation_time = 0
        all_rouge1_scores = []
        all_rouge2_scores = []
        all_rougeL_scores = []
        all_semantic_scores = []
        all_length_ratios = []
        
        for i, test_example in enumerate(self.test_data):
            prompt = test_example["prompt"]
            # Handle both 'answer' and 'expected_output' keys for compatibility
            reference = test_example.get("answer", test_example.get("expected_output", ""))
            
            print(f"   Testing example {i+1}/{len(self.test_data)}: {prompt[:50]}...")
            
            try:
                generated_text, generation_time = self.generate_response(prompt)
                metrics = self.calculate_metrics(generated_text, reference)
                
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
                if "rouge1_f" in metrics:
                    all_rouge1_scores.append(metrics["rouge1_f"])
                if "rouge2_f" in metrics:
                    all_rouge2_scores.append(metrics["rouge2_f"])
                if "rougeL_f" in metrics:
                    all_rougeL_scores.append(metrics["rougeL_f"])
                if "semantic_similarity" in metrics:
                    all_semantic_scores.append(metrics["semantic_similarity"])
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
            "average_rouge1": np.mean(all_rouge1_scores) if all_rouge1_scores else None,
            "average_rouge2": np.mean(all_rouge2_scores) if all_rouge2_scores else None,
            "average_rougeL": np.mean(all_rougeL_scores) if all_rougeL_scores else None,
            "average_semantic_similarity": np.mean(all_semantic_scores) if all_semantic_scores else None,
            "average_length_ratio": np.mean(all_length_ratios) if all_length_ratios else None
        }
        
        print("✅ Evaluation completed!")
    
    def save_results(self):
        """Save evaluation results to file"""
        results_path = f"../UserFacing/db/test_results_cuda_{self.results['test_id']}.json"
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
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        metrics = self.results["summary_metrics"]
        
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Successful Generations: {metrics['successful_generations']}")
        print(f"Success Rate: {metrics['successful_generations']/metrics['total_examples']*100:.1f}%")
        print(f"Average Generation Time: {metrics['average_generation_time']:.3f}s")
        
        if metrics['average_rouge1']:
            print(f"\nROUGE Scores:")
            print(f"  ROUGE-1: {metrics['average_rouge1']:.3f}")
            print(f"  ROUGE-2: {metrics['average_rouge2']:.3f}")
            print(f"  ROUGE-L: {metrics['average_rougeL']:.3f}")
        
        if metrics['average_semantic_similarity']:
            print(f"Semantic Similarity: {metrics['average_semantic_similarity']:.3f}")
        
        if metrics['average_length_ratio']:
            print(f"Average Length Ratio: {metrics['average_length_ratio']:.3f}")
        
        print("\n" + "="*60)

def main():
    evaluator = ModelEvaluator()
    
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
