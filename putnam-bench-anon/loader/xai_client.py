"""
xAI model loader implementation.
Handles API calls to xAI Grok models using OpenAI-compatible interface.
"""

import os
from typing import Dict, Optional, List, Tuple

from .openai_client import OpenAIModelLoader


class XAIModelLoader(OpenAIModelLoader):
    """xAI implementation using OpenAI-compatible API."""
    
    def __init__(self, 
                 solver_model: str = "grok-3",
                 grader_model: str = "grok-3",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize xAI model loader.
        
        Args:
            solver_model: xAI model for solving problems (default: grok-3)
            grader_model: xAI model for grading solutions (default: grok-3)
            api_key: xAI API key (if None, uses XAI_API_KEY environment variable)
            **kwargs: Additional arguments passed to parent class
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('XAI_API_KEY')
        
        # Initialize with xAI-specific settings
        super().__init__(
            solver_model=solver_model,
            grader_model=grader_model,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            **kwargs
        )
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to xAI with proper error handling.
        
        Args:
            model: xAI model name
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Call parent's implementation
            return await super()._call_api(model, messages, temperature)
            
        except Exception as e:
            # Replace "OpenAI" with "xAI" in error messages
            error_msg = str(e)
            if "OpenAI API Error" in error_msg:
                error_msg = error_msg.replace("OpenAI API Error", "xAI API Error")
            
            # Log with xAI-specific prefix
            if "RateLimitError" in type(e).__name__:
                print(f"🚫 xAI RateLimitError: {error_msg}")
                raise
            elif "APIError" in type(e).__name__ or "APIConnectionError" in type(e).__name__:
                print(f"❌ xAI API Error: {error_msg}")
                raise
            else:
                print(f"❌ Unexpected error in xAI API call: {error_msg}")
                raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "xai",
            "base_url": "https://api.x.ai/v1"
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify xAI API connectivity.
        
        Returns:
            True if API is accessible, False otherwise
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
            
            if result and "ok" in result.lower():
                print(f"✅ xAI API health check passed for {self.solver_model}")
                return True
            else:
                print(f"⚠️ xAI API health check returned unexpected response")
                return False
                
        except Exception as e:
            print(f"❌ xAI API health check failed: {str(e)}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate the cost for processing a given number of problems with xAI models.
        
        Args:
            num_problems: Number of problems to process
            avg_problem_length: Average length of problem statements in characters
            avg_solution_length: Average length of solutions in characters
            
        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimates (1 token ≈ 4 characters for English)
        tokens_per_solve = (avg_problem_length + avg_solution_length) // 4
        tokens_per_grade = (avg_problem_length + avg_solution_length * 2) // 4
        
        # xAI pricing (update with actual pricing when available)
        # These are estimates based on similar model pricing
        pricing = {
            "grok-3": {"input": 0.01, "output": 0.03},  # per 1K tokens (estimated)
            "grok-2": {"input": 0.005, "output": 0.015},  # per 1K tokens (estimated)
        }
        
        def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            if model not in pricing:
                model = "grok-3"  # Default to grok-3 pricing
            
            input_cost = (input_tokens / 1000) * pricing[model]["input"]
            output_cost = (output_tokens / 1000) * pricing[model]["output"]
            return input_cost + output_cost
        
        # Calculate costs
        solve_cost = get_model_cost(
            self.solver_model, 
            tokens_per_solve * num_problems,
            tokens_per_solve * num_problems // 2  # Assume output is ~50% of input
        )
        
        grade_cost = get_model_cost(
            self.grader_model,
            tokens_per_grade * num_problems,
            tokens_per_grade * num_problems // 3  # Assume output is ~33% of input
        )
        
        total_cost = solve_cost + grade_cost
        
        return {
            "solve_cost": round(solve_cost, 4),
            "grade_cost": round(grade_cost, 4),
            "total_cost": round(total_cost, 4),
            "cost_per_problem": round(total_cost / num_problems, 6),
            "currency": "USD",
            "note": "xAI pricing estimates - update with actual pricing"
        } 