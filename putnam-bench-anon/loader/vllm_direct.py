"""
VLLM direct Python API model loader implementation.
Uses VLLM's Python API directly without requiring a separate server process.
"""

import asyncio
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

from .base import ModelLoader
from .prompts import SOLVER_SYSTEM_PROMPT, PROOF_GRADER_SYSTEM_PROMPT


class VLLMDirectModelLoader(ModelLoader):
    """VLLM direct Python API implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "gpt2",
                 grader_model: str = "gpt2",
                 max_model_len: int = 512,
                 gpu_memory_utilization: float = 0.4,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize VLLM direct model loader.
        
        Args:
            solver_model: Model name for solving problems (default: gpt2)
            grader_model: Model name for grading solutions (default: gpt2)
            max_model_len: Maximum sequence length (default: 512 for testing)
            gpu_memory_utilization: GPU memory utilization ratio (default: 0.4)
            device: Device to use ('auto', 'cuda', 'cpu')
            **kwargs: Additional arguments passed to parent class
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm package is required for VLLMDirectModelLoader. "
                "Install with: pip install vllm"
            )
            
        super().__init__(solver_model, grader_model, **kwargs)
        
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.device = device
        
        # Model instances (lazy loaded)
        self._solver_llm = None
        self._grader_llm = None
        self._loaded_models = []
        
        print(f"🔧 VLLM Direct loader initialized")
        print(f"   Device: {device}")
        print(f"   Max length: {max_model_len}")
        print(f"   GPU utilization: {gpu_memory_utilization}")
    
    def _get_vllm_config(self, model: str) -> Dict[str, Any]:
        """Get VLLM configuration for a model."""
        return {
            "model": model,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": False,
            "enforce_eager": True,  # Disable graph optimization for faster startup
        }
    
    async def _load_model(self, model: str, purpose: str) -> LLM:
        """Load a VLLM model instance."""
        print(f"📥 Loading {purpose} model: {model}")
        
        try:
            config = self._get_vllm_config(model)
            llm = LLM(**config)
            
            self._loaded_models.append(model)
            print(f"✅ Model loaded successfully: {model}")
            return llm
            
        except Exception as e:
            print(f"❌ Failed to load model {model}: {e}")
            raise
    
    async def _get_solver_model(self) -> LLM:
        """Get or load the solver model."""
        if self._solver_llm is None:
            self._solver_llm = await self._load_model(self.solver_model, "solver")
        return self._solver_llm
    
    async def _get_grader_model(self) -> LLM:
        """Get or load the grader model."""
        if self._grader_llm is None:
            # If solver and grader use the same model, reuse the instance
            if self.solver_model == self.grader_model and self._solver_llm is not None:
                print(f"♻️ Reusing solver model for grading: {self.grader_model}")
                self._grader_llm = self._solver_llm
            else:
                self._grader_llm = await self._load_model(self.grader_model, "grader")
        return self._grader_llm
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        if not messages[-1]["role"] == "assistant":
            prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from model response."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # If no JSON found, try to parse the entire response
            return json.loads(response.strip())
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return None
            return None
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an inference call using VLLM.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Get the appropriate model instance
            if model == self.solver_model:
                llm = await self._get_solver_model()
            elif model == self.grader_model:
                llm = await self._get_grader_model()
            else:
                raise ValueError(f"Unknown model: {model}")
            
            # Convert messages to prompt
            prompt = self._format_messages_as_prompt(messages)
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=500,  # Reasonable limit for responses
                stop=["\nUser:", "\nSystem:"]  # Stop at new conversation turns
            )
            
            # Generate response
            outputs = llm.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text.strip(), generated_text
            else:
                return None, ""
            
        except Exception as e:
            print(f"❌ VLLM inference error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "vllm_direct",
            "device": self.device,
            "loaded_models": self._loaded_models
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify VLLM functionality.
        
        Returns:
            True if models can be loaded and generate text, False otherwise
        """
        try:
            print(f"🔍 VLLM health check starting...")
            
            # Try to load and use the solver model
            test_messages = [
                {"role": "user", "content": "Hello! Please respond with 'Health check OK'."}
            ]
            
            result, _ = await self._call_api(
                model=self.solver_model,
                messages=test_messages,
                temperature=0.1
            )
            
            if result and len(result) > 0:
                print(f"✅ VLLM health check passed for {self.solver_model}")
                print(f"   Response: {result[:50]}...")
                return True
            else:
                print(f"❌ VLLM health check failed: empty response")
                return False
                
        except Exception as e:
            print(f"❌ VLLM health check failed: {str(e)}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate the cost for processing a given number of problems.
        For direct VLLM, cost is computational (time/energy).
        
        Args:
            num_problems: Number of problems to process
            avg_problem_length: Average length of problem statements in characters
            avg_solution_length: Average length of solutions in characters
            
        Returns:
            Dictionary with cost estimates
        """
        # Token estimates (1 token ≈ 4 characters)
        tokens_per_solve = (avg_problem_length + avg_solution_length) // 4
        tokens_per_grade = (avg_problem_length + avg_solution_length * 2) // 4
        
        # Model size cost factors (based on parameter count)
        model_costs = {
            "gpt2": 1.0,      # 124M params
            "distilgpt2": 0.5, # 82M params
            "microsoft/dialo": 1.2,  # DialoGPT variants
            "tinyllama": 2.0,  # 1.1B params
        }
        
        def get_model_cost(model: str) -> float:
            model_lower = model.lower()
            for key, cost in model_costs.items():
                if key in model_lower:
                    return cost
            return 1.5  # Default cost
        
        solver_cost_factor = get_model_cost(self.solver_model)
        grader_cost_factor = get_model_cost(self.grader_model)
        
        # Computational cost estimation (arbitrary units)
        solve_cost = tokens_per_solve * num_problems * solver_cost_factor / 10000
        grade_cost = tokens_per_grade * num_problems * grader_cost_factor / 10000
        
        total_cost = solve_cost + grade_cost
        
        return {
            "solve_cost": round(solve_cost, 4),
            "grade_cost": round(grade_cost, 4),
            "total_cost": round(total_cost, 4),
            "cost_per_problem": round(total_cost / num_problems, 6),
            "currency": "computational_units",
            "note": "Direct VLLM costs are computational (GPU time/energy)"
        }
    
    async def unload_all_models(self):
        """Unload all loaded models to free GPU memory."""
        try:
            print("🗑️ Unloading VLLM models...")
            
            # Clean up model instances
            if self._solver_llm is not None:
                del self._solver_llm
                self._solver_llm = None
            
            if self._grader_llm is not None and self._grader_llm != self._solver_llm:
                del self._grader_llm
                self._grader_llm = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._loaded_models.clear()
            print("✅ Models unloaded successfully")
            
        except Exception as e:
            print(f"⚠️ Error during model cleanup: {e}") 