"""
OpenAI model loader implementation.
Handles API calls to OpenAI models with proper error handling and retry logic.
"""

import asyncio
import random
from typing import Dict, List, Tuple, Optional
import os # Added for KimiModelLoader

from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError, BadRequestError

from .base import ModelLoader
from .prompts import RESPONSE_FORMAT


class OpenAIModelLoader(ModelLoader):
    """OpenAI implementation of the ModelLoader."""
    
    def __init__(self, 
                 solver_model: str = "gpt-4o-mini",
                 grader_model: str = "o3",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenAI model loader.
        
        Args:
            solver_model: OpenAI model for solving problems (default: gpt-4o-mini)
            grader_model: OpenAI model for grading solutions (default: o3)
            api_key: OpenAI API key (if None, uses environment variable)
            base_url: Custom base URL for OpenAI API
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(solver_model, grader_model, **kwargs)
        
        # Initialize OpenAI client with custom httpx client for high concurrency
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        
        # Configure httpx for high concurrency
        import httpx
        limits = httpx.Limits(
            max_connections=1000,  # Total connection pool size
            max_keepalive_connections=500,  # Persistent connections
            keepalive_expiry=30.0  # Keep connections alive for 30s
        )
        timeout = httpx.Timeout(
            timeout=600.0,  # Overall timeout (increased from 300)
            connect=60.0,   # Connection timeout
            read=600.0,     # Read timeout (increased from 300)
            write=60.0      # Write timeout
        )
        
        http_client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout
        )
        client_kwargs["http_client"] = http_client
            
        self.client = AsyncOpenAI(**client_kwargs)
        self._http_client = http_client  # Keep reference to close later
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close http client."""
        if hasattr(self, '_http_client'):
            await self._http_client.aclose()
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to OpenAI.
        
        Args:
            model: OpenAI model name
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Override temperature for models that require it
            # o1, o3, o3-mini, and o4-mini only support temperature 1.0
            if any(model_name in model.lower() for model_name in ['o1', 'o3', 'o3-mini', 'o4-mini']):
                actual_temperature = 1.0
                if self.debug and temperature != 1.0:
                    print(f"⚠️  Overriding temperature from {temperature} to 1.0 for model {model}")
            else:
                actual_temperature = temperature
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": actual_temperature,
                # Set max_tokens to avoid truncation
                # Most OpenAI models support at least 4096, newer ones support much more
                "max_tokens": 32000,  # High default that works for GPT-4 and newer models
            }
            
            # Only add response_format for models that support it
            # o1 models and some older models don't support JSON format
            # Note: o3 and o3-mini DO support response_format (tested and confirmed)
            if not (model.startswith("o1") or model in ["gpt-4", "gpt-3.5-turbo"]):
                api_params["response_format"] = RESPONSE_FORMAT
            
            # Remove max_tokens for models that don't support it
            # o1 and o3 models don't support max_tokens parameter
            if model.startswith("o1") or model.startswith("o3"):
                api_params.pop("max_tokens", None)
            
            # Make the API call
            response = await self.client.chat.completions.create(**api_params)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            return content, content
            
        except RateLimitError as e:
            # Handle rate limiting with special logic
            error_str = str(e)
            if self.debug:
                print(f"🚫 RateLimitError: {error_str}")
            
            if "insufficient_quota" in error_str:
                if self.debug:
                    print("⏳ Detected quota exhaustion - sleeping 15 minutes")
                await asyncio.sleep(900)  # 15 minutes
            else:
                # Standard rate limit - shorter sleep
                sleep_time = 2 + random.random()
                if self.debug:
                    print(f"   ⏰ Rate limited, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
            
            # Re-raise to trigger retry logic
            raise
            
        except BadRequestError as e:
            # Handle policy violations and other 400 errors with special logic
            error_str = str(e)
            if self.debug:
                print(f"🚫 BadRequestError: {error_str}")
            
            if "usage policy" in error_str or "flagged" in error_str:
                if self.debug:
                    print("⏳ Detected policy violation - sleeping 30 seconds before retry")
                await asyncio.sleep(30)  # Longer delay for policy violations
            else:
                # Standard bad request - shorter sleep
                sleep_time = 5 + random.random()
                if self.debug:
                    print(f"   ⏰ Bad request error, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
            
            # Re-raise to trigger retry logic
            raise
            
        except (APIError, APIConnectionError) as e:
            if self.debug:
                print(f"❌ OpenAI API Error: {str(e)}")
            raise
            
        except Exception as e:
            if self.debug:
                print(f"❌ Unexpected error in OpenAI API call: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "openai"
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
            
            # Set temperature based on model
            # o1, o3, o3-mini, and o4-mini require temperature 1.0
            if any(model_name in self.solver_model.lower() for model_name in ['o1', 'o3', 'o3-mini', 'o4-mini']):
                temperature = 1.0
            else:
                # Use temperature 0.0 for deterministic results with other models
                temperature = 0.0
            
            result, _ = await self._call_api(
                model=self.solver_model,
                messages=test_messages,
                temperature=temperature
            )
            
            if result and "ok" in result.lower():
                if self.debug:
                    print(f"✅ OpenAI API health check passed for {self.solver_model}")
                return True
            else:
                if self.debug:
                    print(f"⚠️ OpenAI API health check returned unexpected response")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"❌ OpenAI API health check failed: {str(e)}")
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
        
        # Simplified pricing (update with actual OpenAI pricing)
        # These are rough estimates and should be updated with current pricing
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
            "o3": {"input": 0.03, "output": 0.12},  # per 1K tokens (estimated)
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        }
        
        def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            if model not in pricing:
                model = "gpt-4"  # Default fallback
            
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


class KimiModelLoader(OpenAIModelLoader):
    """Kimi/Moonshot implementation using OpenAI-compatible API."""
    
    def __init__(self, 
                 solver_model: str = "kimi-k2-0711-preview",
                 grader_model: str = "kimi-k2-0711-preview",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize Kimi model loader.
        
        Args:
            solver_model: Kimi model for solving problems (default: moonshot-v1-8k)
            grader_model: Kimi model for grading solutions (default: moonshot-v1-8k)
            api_key: Kimi API key (if None, uses MOONSHOT_API_KEY environment variable)
            **kwargs: Additional arguments passed to parent class
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('MOONSHOT_API_KEY')
        
        # Initialize with Kimi-specific settings
        super().__init__(
            solver_model=solver_model,
            grader_model=grader_model,
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
            **kwargs
        )
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to Kimi with proper error handling.
        
        Args:
            model: Kimi model name
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        import time
        
        start_time = time.time()
        if self.debug:
            print(f"🔄 Starting Kimi API call with model: {model}")
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": RESPONSE_FORMAT,  # Kimi supports JSON format
            }
            
            # Set max_tokens based on model
            if "128k" in model:
                api_params["max_tokens"] = 32000  # For 128k context models
            elif "32k" in model:
                api_params["max_tokens"] = 16000  # For 32k context models  
            elif "8k" in model:
                api_params["max_tokens"] = 8000   # For 8k context models
            elif "k2" in model.lower():
                api_params["max_tokens"] = 24000  # For K2 models
            else:
                api_params["max_tokens"] = 16000  # Default high limit
            
            if self.debug:
                print(f"📋 API call parameters: model={model}, messages={len(messages)}, temp={temperature}, max_tokens={api_params['max_tokens']}")
            
            # Make the API call
            response = await self.client.chat.completions.create(**api_params)
            
            elapsed_time = time.time() - start_time
            if self.debug:
                print(f"✅ Kimi API call completed in {elapsed_time:.2f}s")
            
            # Extract response content
            content = response.choices[0].message.content or ""
            if self.debug:
                print(f"📄 Response length: {len(content)} characters")
            
            # Check if response might be truncated
            if self.debug and hasattr(response, 'usage'):
                completion_tokens = response.usage.completion_tokens
                print(f"📊 Completion tokens used: {completion_tokens}")
                if completion_tokens >= api_params['max_tokens'] * 0.95:  # 95% of limit
                    print(f"⚠️  WARNING: Response may be truncated (used {completion_tokens}/{api_params['max_tokens']} tokens)")
            
            # Check if content ends abruptly (truncation signs)
            if self.debug and content and not content.strip().endswith(('"}', '"}')):
                print("⚠️  WARNING: Response doesn't end with proper JSON closure - likely truncated")
            
            # ============= RAW RESPONSE LOGGING (DEBUG ONLY) =============
            if self.debug:
                import json
                from pathlib import Path
                from datetime import datetime
                
                # Create raw response log directory
                log_dir = Path("kimi_raw_responses")
                log_dir.mkdir(exist_ok=True)
                
                # Save raw response
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
                raw_log_file = log_dir / f"kimi_raw_response_{timestamp}.json"
                
                raw_response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "api_params": api_params,
                    "response_time_seconds": elapsed_time,
                    "raw_content": content,
                    "content_length": len(content),
                    "response_object": {
                        "choices": [
                            {
                                "message": {
                                    "content": content,
                                    "role": response.choices[0].message.role
                                }
                            }
                        ]
                    }
                }
                
                try:
                    with open(raw_log_file, 'w', encoding='utf-8') as f:
                        json.dump(raw_response_data, f, indent=2, ensure_ascii=False)
                    print(f"💾 Raw response saved to: {raw_log_file}")
                except Exception as save_error:
                    print(f"❌ Failed to save raw response: {save_error}")
                
                # Also print raw content to console
                print(f"📋 RAW RESPONSE CONTENT:")
                print(f"{'='*60}")
                print(content[:1000] + ("..." if len(content) > 1000 else ""))
                print(f"{'='*60}")
            # ============= END RAW RESPONSE LOGGING =============
            
            return content, content
            
        except RateLimitError as e:
            elapsed_time = time.time() - start_time
            error_str = str(e)
            if self.debug:
                print(f"🚫 Kimi RateLimitError after {elapsed_time:.2f}s: {error_str}")
            
            # Try to capture response details
            if self.debug and hasattr(e, 'response') and e.response:
                print(f"   Status: {e.response.status_code}")
                print(f"   Headers: {dict(e.response.headers)}")
                print(f"   Response: {e.response.text[:500]}...")
            
            if "insufficient_quota" in error_str:
                if self.debug:
                    print("⏳ Detected Kimi quota exhaustion - sleeping 15 minutes")
                await asyncio.sleep(900)  # 15 minutes
            else:
                # Standard rate limit - shorter sleep
                sleep_time = 2 + random.random()
                if self.debug:
                    print(f"   ⏰ Rate limited on Kimi API, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
            
            # Re-raise to trigger retry logic
            raise
            
        except (APIError, APIConnectionError) as e:
            elapsed_time = time.time() - start_time
            error_str = str(e)
            if self.debug:
                print(f"❌ Kimi API Error after {elapsed_time:.2f}s: {error_str}")
            
            # Try to capture response details
            if self.debug and hasattr(e, 'response') and e.response:
                print(f"   Status: {e.response.status_code}")
                print(f"   Headers: {dict(e.response.headers)}")
                print(f"   Response: {e.response.text[:500]}...")
            
            # Log request details for debugging
            if self.debug and hasattr(e, 'request') and e.request:
                print(f"   Request URL: {e.request.url}")
                print(f"   Request method: {e.request.method}")
                print(f"   Request headers: {dict(e.request.headers)}")
            
            raise
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_str = str(e)
            if self.debug:
                print(f"❌ Unexpected error in Kimi API call after {elapsed_time:.2f}s: {error_str}")
                print(f"   Error type: {type(e).__name__}")
            
            # Try to capture any additional error details
            if self.debug and hasattr(e, 'response'):
                try:
                    print(f"   Response status: {e.response.status_code}")
                    print(f"   Response headers: {dict(e.response.headers)}")
                    print(f"   Response text: {e.response.text[:500]}...")
                except:
                    print("   Could not extract response details")
            
            # Log the full exception
            if self.debug:
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
            
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "kimi",
            "base_url": "https://api.moonshot.ai/v1"
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify Kimi API connectivity.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test call with Kimi's system prompt
            test_messages = [
                {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
                {"role": "user", "content": "Hello, please respond with a simple JSON: {\"status\": \"ok\"}"}
            ]
            
            result, _ = await self._call_api(
                model=self.solver_model,
                messages=test_messages,
                temperature=0.0
            )
            
            if result and "ok" in result.lower():
                if self.debug:
                    print(f"✅ Kimi API health check passed for {self.solver_model}")
                return True
            else:
                if self.debug:
                    print(f"⚠️ Kimi API health check returned unexpected response")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"❌ Kimi API health check failed: {str(e)}")
            return False
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate the cost for processing a given number of problems with Kimi models.
        
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
        
        # Kimi pricing (in USD per 1K tokens)
        # These are example prices - update with actual Kimi pricing
        pricing = {
            "moonshot-v1-8k": {"input": 0.012, "output": 0.012},
            "moonshot-v1-32k": {"input": 0.024, "output": 0.024},
            "moonshot-v1-128k": {"input": 0.06, "output": 0.06},
        }
        
        def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            if model not in pricing:
                model = "moonshot-v1-8k"  # Default to 8k pricing
            
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
            tokens_per_grade * num_problems // 4  # Grading output is shorter
        )
        
        return {
            "solver_cost": solve_cost,
            "grader_cost": grade_cost,
            "total_cost": solve_cost + grade_cost,
            "num_problems": num_problems,
            "solver_model": self.solver_model,
            "grader_model": self.grader_model
        }
