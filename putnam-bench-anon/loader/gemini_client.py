"""
Gemini model loader implementation.
Handles API calls to Google Gemini models with proper error handling and retry logic.
"""

import asyncio
import random
from typing import Dict, List, Tuple, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import generation_types
except ImportError:
    genai = None
    generation_types = None

from .base import ModelLoader
from .prompts import RESPONSE_FORMAT


class GeminiModelLoader(ModelLoader):
    """Gemini implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "gemini-1.5-flash",
                 grader_model: str = "gemini-1.5-pro",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize Gemini model loader.
        
        Args:
            solver_model: Gemini model for solving problems (default: gemini-1.5-flash)
            grader_model: Gemini model for grading solutions (default: gemini-1.5-pro)
            api_key: Google AI API key (if None, uses environment variable GOOGLE_API_KEY)
            **kwargs: Additional arguments passed to parent class
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required for GeminiModelLoader. "
                "Install with: pip install google-generativeai"
            )
            
        super().__init__(solver_model, grader_model, **kwargs)
        
        # Configure Google AI
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Will use GOOGLE_API_KEY environment variable
            genai.configure()
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to Gemini.
        
        Args:
            model: Gemini model name
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Initialize the model
            model_instance = genai.GenerativeModel(model)
            
            # Convert OpenAI format to Gemini format
            system_instruction = None
            conversation = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    conversation.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    conversation.append({"role": "model", "parts": [msg["content"]]})
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4000,
            )
            
            # Request JSON format for all Gemini models
            # Flash models now support JSON format as per latest API documentation
            generation_config.response_mime_type = "application/json"
            
            # Make the API call
            if system_instruction and len(conversation) == 1:
                # Single user message with system instruction
                prompt = f"{system_instruction}\n\n{conversation[0]['parts'][0]}"
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    prompt,
                    generation_config=generation_config
                )
            else:
                # Multi-turn conversation
                if system_instruction:
                    # Prepend system instruction to first user message
                    if conversation and conversation[0]["role"] == "user":
                        conversation[0]["parts"][0] = f"{system_instruction}\n\n{conversation[0]['parts'][0]}"
                
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    conversation,
                    generation_config=generation_config
                )
            
            # Extract response content
            content = ""
            if response.text:
                content = response.text
            
            return content, content
            
        except Exception as e:
            error_str = str(e)
            
            # Handle different types of errors
            if "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"🚫 Rate/Quota Error: {error_str}")
                if "quota" in error_str.lower():
                    print("⏳ Detected quota exhaustion - sleeping 15 minutes")
                    await asyncio.sleep(900)  # 15 minutes
                else:
                    sleep_time = 2 + random.random()
                    print(f"   ⏰ Rate limited, sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                # Re-raise to trigger retry logic
                raise
            elif "api" in error_str.lower():
                print(f"❌ Gemini API Error: {error_str}")
                raise
            else:
                print(f"❌ Unexpected error in Gemini API call: {error_str}")
                raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "gemini"
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
                print(f"✅ Gemini API health check passed for {self.solver_model}")
                return True
            else:
                print(f"⚠️ Gemini API health check returned unexpected response")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API health check failed: {str(e)}")
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
        
        # Gemini pricing (update with actual Google AI pricing)
        # These are rough estimates and should be updated with current pricing
        pricing = {
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},  # per 1K tokens
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},  # per 1K tokens
            "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},  # per 1K tokens
        }
        
        def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            if model not in pricing:
                model = "gemini-1.5-pro"  # Default fallback
            
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
