"""
OpenRouter model loader implementation.
Handles API calls to OpenRouter service using OpenAI-compatible interface.
OpenRouter provides access to multiple model providers through a single API.
"""

import os
from typing import Dict, Optional, List, Tuple

from .openai_client import OpenAIModelLoader


class OpenRouterModelLoader(OpenAIModelLoader):
    """OpenRouter implementation using OpenAI-compatible API."""
    
    def __init__(self, 
                 solver_model: str = "openai/gpt-4o",
                 grader_model: str = "openai/gpt-4o",
                 api_key: Optional[str] = None,
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenRouter model loader.
        
        Args:
            solver_model: Model for solving problems (default: openai/gpt-4o)
                        Format should be "provider/model-name" (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
            grader_model: Model for grading solutions (default: openai/gpt-4o)
                        Format should be "provider/model-name"
            api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
            site_url: Optional site URL for rankings on openrouter.ai
            site_name: Optional site name for rankings on openrouter.ai
            **kwargs: Additional arguments passed to parent class
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or pass api_key parameter")
        
        # Store site information for headers
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize with OpenRouter-specific settings
        super().__init__(
            solver_model=solver_model,
            grader_model=grader_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs
        )
    
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to OpenRouter with proper headers.
        
        Args:
            model: Model name in format "provider/model-name"
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        try:
            # Prepare extra headers for OpenRouter
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                # Set max_tokens to avoid truncation, especially for models like Gemini
                # 32000 is a reasonable default that works for most models
                "max_tokens": 32000,
            }
            
            # Add response_format for all models - OpenRouter handles compatibility
            from .prompts import RESPONSE_FORMAT
            api_params["response_format"] = RESPONSE_FORMAT
            
            # Make the API call with extra headers
            if extra_headers:
                response = await self.client.chat.completions.create(
                    **api_params,
                    extra_headers=extra_headers
                )
            else:
                response = await self.client.chat.completions.create(**api_params)
            
            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise ValueError("Empty response from OpenRouter API")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in OpenRouter API response")
                
            return content, content
            
        except Exception as e:
            # Replace "OpenAI" with "OpenRouter" in error messages
            error_msg = str(e)
            if "OpenAI API Error" in error_msg:
                error_msg = error_msg.replace("OpenAI API Error", "OpenRouter API Error")
            
            # Log with OpenRouter-specific prefix
            if "RateLimitError" in type(e).__name__:
                print(f"🚫 OpenRouter RateLimitError: {error_msg}")
                raise
            elif "APIError" in type(e).__name__ or "APIConnectionError" in type(e).__name__:
                print(f"❌ OpenRouter API Error: {error_msg}")
                raise
            else:
                print(f"❌ Unexpected error in OpenRouter API call: {error_msg}")
                raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models."""
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1"
        }
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check to verify OpenRouter API connectivity.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test call
            test_messages = [
                {"role": "user", "content": "Hello, please respond with a simple JSON: {\"status\": \"ok\"}"}
            ]
            
            result, _ = await self._call_api(
                self.solver_model,
                test_messages,
                temperature=0.0
            )
            
            return result is not None
            
        except Exception as e:
            print(f"❌ OpenRouter health check failed: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of commonly available models on OpenRouter.
        Note: This is not exhaustive. Check https://openrouter.ai/models for full list.
        
        Returns:
            List of model identifiers in "provider/model-name" format
        """
        return [
            # OpenAI models
            "openai/gpt-4o",
            "openai/gpt-4o-mini", 
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "openai/o1-preview",
            "openai/o1-mini",
            
            # Anthropic models
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "anthropic/claude-2.1",
            "anthropic/claude-2",
            
            # Google models
            "google/gemini-pro",
            "google/gemini-pro-vision",
            "google/palm-2-codechat-bison",
            "google/palm-2-chat-bison",
            
            # Meta models
            "meta-llama/llama-3-70b-instruct",
            "meta-llama/llama-3-8b-instruct",
            "meta-llama/codellama-70b-instruct",
            
            # Mistral models
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "mistralai/mistral-small",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            
            # Other notable models
            "cohere/command-r-plus",
            "cohere/command-r",
            "databricks/dbrx-instruct",
            "deepseek/deepseek-coder",
            "deepseek/deepseek-chat",
            "qwen/qwen-2-72b-instruct",
            "qwen/qwen-1.5-110b-chat",
        ] 