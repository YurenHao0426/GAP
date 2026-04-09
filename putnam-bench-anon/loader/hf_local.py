"""
Hugging Face local model loader implementation.
Handles direct inference with locally loaded transformers models.
"""

import asyncio
import random
from typing import Dict, List, Tuple, Optional
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import transformers
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    transformers = None

from .base import ModelLoader


class HuggingFaceModelLoader(ModelLoader):
    """Hugging Face local model implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "microsoft/DialoGPT-medium",
                 grader_model: str = "microsoft/DialoGPT-large",
                 device: str = "auto",
                 max_length: int = 4000,
                 **kwargs):
        """
        Initialize Hugging Face model loader.
        
        Args:
            solver_model: HuggingFace model name for solving problems
            grader_model: HuggingFace model name for grading solutions  
            device: Device to run models on ("auto", "cuda", "cpu")
            max_length: Maximum generation length
            **kwargs: Additional arguments passed to parent class
        """
        if transformers is None or torch is None:
            raise ImportError(
                "transformers and torch packages are required for HuggingFaceModelLoader. "
                "Install with: pip install transformers torch"
            )
            
        super().__init__(solver_model, grader_model, **kwargs)
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.max_length = max_length
        
        # Model and tokenizer caches
        self._models = {}
        self._tokenizers = {}
        self._pipelines = {}
        
        print(f"🔧 HuggingFace loader initialized on device: {self.device}")
    
    async def _load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer, with caching."""
        if model_name not in self._models:
            print(f"📥 Loading model: {model_name}")
            
            try:
                # Load in a separate thread to avoid blocking
                tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, 
                    model_name,
                    trust_remote_code=True
                )
                
                model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    model = model.to(self.device)
                
                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer
                
                print(f"✅ Model loaded successfully: {model_name}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {str(e)}")
                raise
        
        return self._models[model_name], self._tokenizers[model_name]
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make a local inference call using the HuggingFace model.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Load model and tokenizer
            hf_model, tokenizer = await self._load_model(model)
            
            # Convert messages to prompt format
            prompt = self._format_messages(messages)
            
            # Generate response
            response = await self._generate_response(
                hf_model, tokenizer, prompt, temperature
            )
            
            return response, response
            
        except Exception as e:
            print(f"❌ HuggingFace inference error: {str(e)}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to a prompt string."""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _generate_response(self, 
                               model: AutoModelForCausalLM,
                               tokenizer: AutoTokenizer,
                               prompt: str, 
                               temperature: float) -> str:
        """Generate response using the loaded model."""
        
        # Tokenize input
        inputs = await asyncio.to_thread(
            tokenizer.encode,
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Leave room for generation
        )
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": min(self.max_length, 2048),
            "temperature": max(temperature, 0.1),  # Avoid 0 temperature
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "attention_mask": torch.ones_like(inputs)
        }
        
        if temperature > 0.0:
            gen_kwargs.update({
                "top_p": 0.9,
                "top_k": 50
            })
        
        # Generate
        with torch.no_grad():
            outputs = await asyncio.to_thread(
                model.generate,
                inputs,
                **gen_kwargs
            )
        
        # Decode response
        generated_text = await asyncio.to_thread(
            tokenizer.decode,
            outputs[0][inputs.shape[1]:],  # Only new tokens
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "huggingface",
            "device": self.device,
            "loaded_models": list(self._models.keys())
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check by testing model loading and inference.
        
        Returns:
            True if models can be loaded and run, False otherwise
        """
        try:
            # Simple test
            test_messages = [
                {"role": "user", "content": "Hello, please say 'ok' to confirm you're working."}
            ]
            
            result, _ = await self._call_api(
                model=self.solver_model,
                messages=test_messages,
                temperature=0.1
            )
            
            if result and len(result) > 0:
                print(f"✅ HuggingFace health check passed for {self.solver_model}")
                return True
            else:
                print(f"⚠️ HuggingFace health check returned empty response")
                return False
                
        except Exception as e:
            print(f"❌ HuggingFace health check failed: {str(e)}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate computational cost for processing problems locally.
        
        Args:
            num_problems: Number of problems to process
            avg_problem_length: Average length of problem statements in characters
            avg_solution_length: Average length of solutions in characters
            
        Returns:
            Dictionary with cost estimates (computational cost in arbitrary units)
        """
        # Rough token estimates (1 token ≈ 4 characters for English)
        tokens_per_solve = (avg_problem_length + avg_solution_length) // 4
        tokens_per_grade = (avg_problem_length + avg_solution_length * 2) // 4
        
        # Model size-based cost estimation (FLOPS approximation)
        model_costs = {
            # Small models (< 1B parameters)
            "gpt2": 0.5,
            "distilgpt2": 0.3,
            "dialogpt-small": 0.4,
            "dialogpt-medium": 0.8,
            
            # Medium models (1B - 10B parameters)  
            "dialogpt-large": 1.5,
            "gpt2-medium": 1.0,
            "gpt2-large": 2.0,
            "gpt2-xl": 4.0,
            
            # Large models (10B+ parameters)
            "llama-7b": 8.0,
            "llama-13b": 15.0,
            "llama-30b": 35.0,
            "llama-65b": 70.0,
        }
        
        def get_model_cost(model: str) -> float:
            model_lower = model.lower()
            for key, cost in model_costs.items():
                if key in model_lower:
                    return cost
            
            # Default based on common model sizes
            if any(size in model_lower for size in ["small", "mini"]):
                return 0.5
            elif any(size in model_lower for size in ["medium", "base"]):
                return 1.0  
            elif any(size in model_lower for size in ["large", "xl"]):
                return 2.0
            else:
                return 1.5  # Default for unknown models
        
        # Calculate computational costs
        solver_cost_factor = get_model_cost(self.solver_model)
        grader_cost_factor = get_model_cost(self.grader_model)
        
        # Device multiplier (GPU is faster but uses more power)
        device_multiplier = 0.3 if self.device == "cuda" else 1.0
        
        solve_cost = tokens_per_solve * num_problems * solver_cost_factor * device_multiplier / 1000
        grade_cost = tokens_per_grade * num_problems * grader_cost_factor * device_multiplier / 1000
        
        total_cost = solve_cost + grade_cost
        
        return {
            "solve_cost": round(solve_cost, 4),
            "grade_cost": round(grade_cost, 4), 
            "total_cost": round(total_cost, 4),
            "cost_per_problem": round(total_cost / num_problems, 6),
            "currency": "computational_units",
            "device": self.device,
            "note": "Local HuggingFace costs are computational (time/energy/memory)"
        }
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model to free memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            if model_name in self._models:
                del self._models[model_name]
                del self._tokenizers[model_name]
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"🗑️ Unloaded model: {model_name}")
                return True
            else:
                print(f"⚠️ Model not loaded: {model_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error unloading model {model_name}: {str(e)}")
            return False
    
    async def unload_all_models(self) -> bool:
        """
        Unload all models to free memory.
        
        Returns:
            True if all models successfully unloaded
        """
        try:
            model_names = list(self._models.keys())
            success = True
            
            for model_name in model_names:
                if not await self.unload_model(model_name):
                    success = False
            
            return success
            
        except Exception as e:
            print(f"❌ Error unloading all models: {str(e)}")
            return False
