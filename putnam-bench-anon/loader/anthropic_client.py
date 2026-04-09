"""
Anthropic model loader implementation.
Handles API calls to Anthropic Claude models with proper error handling and retry logic.
"""

import asyncio
import random
from typing import Dict, List, Tuple, Optional

try:
    from anthropic import AsyncAnthropic, RateLimitError, APIError, APIConnectionError
except ImportError:
    AsyncAnthropic = None
    RateLimitError = Exception
    APIError = Exception
    APIConnectionError = Exception

from .base import ModelLoader
from .prompts import RESPONSE_FORMAT


class AnthropicModelLoader(ModelLoader):
    """Anthropic implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "claude-3-5-haiku-20241022",
                 grader_model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize Anthropic model loader.
        
        Args:
            solver_model: Anthropic model for solving problems (default: claude-3-5-haiku)
            grader_model: Anthropic model for grading solutions (default: claude-3-5-sonnet)
            api_key: Anthropic API key (if None, uses environment variable)
            base_url: Custom base URL for Anthropic API
            **kwargs: Additional arguments passed to parent class
        """
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required for AnthropicModelLoader. "
                "Install with: pip install anthropic"
            )
            
        super().__init__(solver_model, grader_model, **kwargs)
        
        # Initialize Anthropic client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = AsyncAnthropic(**client_kwargs)
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to Anthropic.
        
        Args:
            model: Anthropic model name
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Convert OpenAI format to Anthropic format
            system_message = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": user_messages,
                "max_tokens": 4000,  # Anthropic requires max_tokens
                "temperature": temperature,
            }
            
            if system_message:
                api_params["system"] = system_message
            
            # Make the API call
            response = await self.client.messages.create(**api_params)
            
            # Extract response content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            return content, content
            
        except RateLimitError as e:
            # Handle rate limiting with special logic
            error_str = str(e)
            print(f"🚫 RateLimitError: {error_str}")
            
            if "insufficient_quota" in error_str.lower():
                print("⏳ Detected quota exhaustion - sleeping 15 minutes")
                await asyncio.sleep(900)  # 15 minutes
            else:
                # Standard rate limit - shorter sleep
                sleep_time = 2 + random.random()
                print(f"   ⏰ Rate limited, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
            
            # Re-raise to trigger retry logic
            raise
            
        except (APIError, APIConnectionError) as e:
            print(f"❌ Anthropic API Error: {str(e)}")
            raise
            
        except Exception as e:
            print(f"❌ Unexpected error in Anthropic API call: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "anthropic"
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify API connectivity.
        
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
                print(f"✅ Anthropic API health check passed for {self.solver_model}")
                return True
            else:
                print(f"⚠️ Anthropic API health check returned unexpected response")
                return False
                
        except Exception as e:
            print(f"❌ Anthropic API health check failed: {str(e)}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate the cost for processing a given number of problems.
        
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
        
        # Anthropic pricing (update with actual Anthropic pricing)
        # These are rough estimates and should be updated with current pricing
        pricing = {
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},  # per 1K tokens
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},  # per 1K tokens
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},  # per 1K tokens
        }
        
        def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            if model not in pricing:
                model = "claude-3-5-sonnet-20241022"  # Default fallback
            
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
            "currency": "USD"
        }
