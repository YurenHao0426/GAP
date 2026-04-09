"""
VLLM local model loader implementation.
Handles API calls to locally deployed VLLM services with OpenAI-compatible endpoints.
"""

import asyncio
import random
from typing import Dict, List, Tuple, Optional

try:
    from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
except ImportError:
    AsyncOpenAI = None
    RateLimitError = Exception
    APIError = Exception
    APIConnectionError = Exception

from .base import ModelLoader
from .prompts import RESPONSE_FORMAT


class VLLMModelLoader(ModelLoader):
    """VLLM local model implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 grader_model: str = "meta-llama/Llama-3.2-8B-Instruct", 
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY",
                 **kwargs):
        """
        Initialize VLLM model loader.
        
        Args:
            solver_model: Model name for solving problems (default: Llama-3.2-3B-Instruct)
            grader_model: Model name for grading solutions (default: Llama-3.2-8B-Instruct)
            base_url: VLLM server URL (default: http://localhost:8000/v1)
            api_key: API key for VLLM server (default: "EMPTY" for local)
            **kwargs: Additional arguments passed to parent class
        """
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for VLLMModelLoader. "
                "Install with: pip install openai"
            )
            
        super().__init__(solver_model, grader_model, **kwargs)
        
        # Initialize OpenAI-compatible client for VLLM
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.base_url = base_url
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to VLLM server.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000,
            }
            
            # Only add response_format for models that support it
            # Most local models may not support structured JSON output
            if temperature == 0.0:
                try:
                    api_params["response_format"] = RESPONSE_FORMAT
                except:
                    # If JSON format is not supported, we'll parse manually
                    pass
            
            # Make the API call
            response = await self.client.chat.completions.create(**api_params)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            return content, content
            
        except (RateLimitError, APIError, APIConnectionError) as e:
            # Handle various API errors
            error_str = str(e)
            print(f"❌ VLLM API Error: {error_str}")
            
            if "rate" in error_str.lower() or "limit" in error_str.lower():
                sleep_time = 2 + random.random()
                print(f"   ⏰ Rate limited, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
            
            # Re-raise to trigger retry logic
            raise
            
        except Exception as e:
            print(f"❌ Unexpected error in VLLM API call: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "vllm",
            "base_url": self.base_url
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify VLLM server connectivity.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            # Simple test call
            test_messages = [
                {"role": "user", "content": "Hello, please respond with a simple JSON: {\"status\": \"ok\"}"}
            ]
            
            result, _ = await self._call_api(
                model=self.solver_model,
                messages=test_messages,
                temperature=0.0
            )
            
            if result and ("ok" in result.lower() or "hello" in result.lower()):
                print(f"✅ VLLM API health check passed for {self.solver_model}")
                return True
            else:
                print(f"⚠️ VLLM API health check returned unexpected response")
                return False
                
        except Exception as e:
            print(f"❌ VLLM API health check failed: {str(e)}")
            print(f"   Make sure VLLM server is running at {self.base_url}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate the cost for processing a given number of problems.
        For local VLLM, cost is typically computational (time/energy) rather than monetary.
        
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
        
        # Computational cost estimation (arbitrary units based on model size)
        # Larger models consume more computational resources
        model_costs = {
            "llama-3.2-1b": 1.0,
            "llama-3.2-3b": 2.0, 
            "llama-3.2-8b": 4.0,
            "llama-3.1-8b": 4.0,
            "llama-3.1-70b": 20.0,
            "mistral-7b": 3.0,
            "qwen2.5-7b": 3.0,
        }
        
        def get_model_cost(model: str) -> float:
            model_lower = model.lower()
            for key, cost in model_costs.items():
                if key in model_lower:
                    return cost
            return 3.0  # Default cost for unknown models
        
        # Calculate computational costs
        solver_cost_factor = get_model_cost(self.solver_model)
        grader_cost_factor = get_model_cost(self.grader_model)
        
        solve_cost = tokens_per_solve * num_problems * solver_cost_factor / 1000
        grade_cost = tokens_per_grade * num_problems * grader_cost_factor / 1000
        
        total_cost = solve_cost + grade_cost
        
        return {
            "solve_cost": round(solve_cost, 4),
            "grade_cost": round(grade_cost, 4),
            "total_cost": round(total_cost, 4),
            "cost_per_problem": round(total_cost / num_problems, 6),
            "currency": "computational_units",
            "note": "Local VLLM costs are computational (time/energy) rather than monetary"
        }
    
    async def list_models(self) -> List[str]:
        """
        List available models on the VLLM server.
        
        Returns:
            List of available model names
        """
        try:
            # Try to get models list from VLLM server
            models_response = await self.client.models.list()
            return [model.id for model in models_response.data]
        except Exception as e:
            print(f"⚠️ Could not retrieve models list: {str(e)}")
            return [self.solver_model, self.grader_model]
