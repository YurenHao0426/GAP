"""
Model loader package for mathematical problem solving.

This package provides a unified interface for loading and using different AI models
to solve mathematical problems and grade solutions.

Usage:
    # Create an OpenAI loader
    loader = create_loader("openai", solver_model="gpt-4o-mini", grader_model="o3")
    
    # Create an OpenRouter loader
    loader = create_loader("openrouter", solver_model="openai/gpt-4o", grader_model="anthropic/claude-3-opus")
    
    # Or directly instantiate
    from loader import OpenAIModelLoader
    loader = OpenAIModelLoader()
    
    # Test a problem
    import json
    with open("dataset/1938-A-1.json") as f:
        data = json.load(f)
    
    result = await loader.test_single_problem(data, variant_type="original")
"""

from .base import ModelLoader
from .openai_client import OpenAIModelLoader, KimiModelLoader
from .anthropic_client import AnthropicModelLoader
from .gemini_client import GeminiModelLoader
from .xai_client import XAIModelLoader
from .openrouter_client import OpenRouterModelLoader
from .vllm_local import VLLMModelLoader
from .vllm_direct import VLLMDirectModelLoader
from .hf_local import HuggingFaceModelLoader
from .cross_provider import CrossProviderLoader
from .prompts import (
    SOLVER_SYSTEM_PROMPT,
    SOLVER_USER_TEMPLATE,
    PROOF_GRADER_SYSTEM_PROMPT,
    CALCULATION_GRADER_SYSTEM_PROMPT,
    PROOF_GRADER_USER_TEMPLATE,
    CALCULATION_GRADER_USER_TEMPLATE,
    RESPONSE_FORMAT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_BASE
)
from typing import Optional

def create_loader(provider: str, **kwargs) -> ModelLoader:
    """
    Factory function to create model loaders.
    
    Args:
        provider: Provider name ("openai", "anthropic", "gemini", etc.)
        **kwargs: Additional arguments passed to the loader constructor
        
    Returns:
        ModelLoader instance
        
    Raises:
        ValueError: If provider is not supported
        
    Examples:
        # Create OpenAI loader
        loader = create_loader("openai", solver_model="gpt-4o-mini", grader_model="o3")
        
        # Create loader with custom settings
        loader = create_loader(
            "openai", 
            solver_model="gpt-4", 
            grader_model="o3",
            retries=5,
            timeout_base=600
        )
    """
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return OpenAIModelLoader(**kwargs)
    elif provider_lower == "anthropic":
        return AnthropicModelLoader(**kwargs)
    elif provider_lower == "gemini":
        return GeminiModelLoader(**kwargs)
    elif provider_lower == "xai":
        return XAIModelLoader(**kwargs)
    elif provider_lower == "openrouter":
        return OpenRouterModelLoader(**kwargs)
    elif provider_lower == "kimi":
        return KimiModelLoader(**kwargs)
    elif provider_lower == "vllm":
        return VLLMModelLoader(**kwargs)
    elif provider_lower == "vllm_direct":
        return VLLMDirectModelLoader(**kwargs)
    elif provider_lower in ["huggingface", "hf"]:
        return HuggingFaceModelLoader(**kwargs)
    else:
        supported = ["openai", "anthropic", "gemini", "xai", "openrouter", "kimi", "vllm", "vllm_direct", "huggingface"]
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {supported}")


def create_cross_provider_loader(
    solver_provider: str,
    grader_provider: Optional[str] = None,
    solver_model: Optional[str] = None,
    grader_model: Optional[str] = None,
    **kwargs
) -> ModelLoader:
    """
    Create a loader that can use different providers for solving and grading.
    
    Args:
        solver_provider: Provider for solving problems
        grader_provider: Provider for grading (if None, uses solver_provider)
        solver_model: Override solver model
        grader_model: Override grader model
        **kwargs: Additional arguments (can include provider-specific settings)
        
    Returns:
        CrossProviderLoader instance
        
    Examples:
        # Use Kimi for solving and OpenAI for grading
        loader = create_cross_provider_loader(
            solver_provider="kimi",
            grader_provider="openai",
            solver_model="Kimi-K2-Instruct",
            grader_model="o3"
        )
        
        # Use same provider but different models
        loader = create_cross_provider_loader(
            solver_provider="openai",
            solver_model="gpt-4o-mini",
            grader_model="o3"
        )
    """
    # Extract provider-specific kwargs
    solver_kwargs = kwargs.pop('solver_kwargs', {})
    grader_kwargs = kwargs.pop('grader_kwargs', {})
    
    # Extract common parameters that should be passed to both loaders
    quick = kwargs.pop('quick', False)
    debug = kwargs.pop('debug', False)
    
    # Add common parameters to both solver and grader kwargs
    solver_kwargs.update({'quick': quick, 'debug': debug})
    grader_kwargs.update({'quick': quick, 'debug': debug})
    
    # Get default models if not specified
    if not solver_model:
        solver_defaults = get_default_models(solver_provider)
        solver_model = solver_defaults['solver_model']
    
    if not grader_provider:
        grader_provider = solver_provider
        
    if not grader_model:
        grader_defaults = get_default_models(grader_provider)
        grader_model = grader_defaults['grader_model']
    
    # Create solver loader
    solver_loader = create_loader(
        solver_provider,
        solver_model=solver_model,
        grader_model=solver_model,  # Use solver model for both in solver loader
        **solver_kwargs
    )
    
    # Create grader loader if different provider
    if grader_provider != solver_provider:
        grader_loader = create_loader(
            grader_provider,
            solver_model=grader_model,  # Use grader model for both in grader loader
            grader_model=grader_model,
            **grader_kwargs
        )
        return CrossProviderLoader(solver_loader, grader_loader, **kwargs)
    else:
        # Same provider, but possibly different models
        if solver_model != grader_model:
            # Need to create a single loader with both models
            single_loader = create_loader(
                solver_provider,
                solver_model=solver_model,
                grader_model=grader_model,
                **solver_kwargs
            )
            return single_loader
        else:
            # Same provider and model
            return solver_loader

def get_supported_providers() -> list[str]:
    """
    Get list of supported model providers.
    
    Returns:
        List of supported provider names
    """
    return ["openai", "anthropic", "gemini", "xai", "openrouter", "kimi", "vllm", "vllm_direct", "huggingface"]

def get_default_models(provider: str) -> dict[str, str]:
    """
    Get default model names for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Dictionary with default solver_model and grader_model
    """
    defaults = {
        "openai": {
            "solver_model": "gpt-4o-mini",
            "grader_model": "o3"
        },
        "anthropic": {
            "solver_model": "claude-3-5-haiku-20241022",
            "grader_model": "claude-3-5-sonnet-20241022"
        },
        "gemini": {
            "solver_model": "gemini-1.5-flash",
            "grader_model": "gemini-1.5-pro"
        },
        "xai": {
            "solver_model": "grok-3",
            "grader_model": "grok-3"
        },
        "openrouter": {
            "solver_model": "openai/gpt-4o",
            "grader_model": "openai/gpt-4o"
        },
        "kimi": {
            "solver_model": "moonshot-v1-8k",
            "grader_model": "moonshot-v1-8k"
        },
        "vllm": {
            "solver_model": "meta-llama/Llama-3.2-3B-Instruct",
            "grader_model": "meta-llama/Llama-3.2-8B-Instruct"
        },
        "vllm_direct": {
            "solver_model": "gpt2",
            "grader_model": "gpt2"
        },
        "huggingface": {
            "solver_model": "microsoft/DialoGPT-medium",
            "grader_model": "microsoft/DialoGPT-large"
        }
    }
    
    provider_lower = provider.lower()
    if provider_lower not in defaults:
        raise ValueError(f"No defaults available for provider: {provider}")
    
    return defaults[provider_lower]

# Export main classes and functions
__all__ = [
    # Main classes
    "ModelLoader",
    "OpenAIModelLoader",
    "AnthropicModelLoader",
    "GeminiModelLoader",
    "XAIModelLoader",
    "OpenRouterModelLoader",
    "KimiModelLoader",
    "VLLMModelLoader",
    "VLLMDirectModelLoader",
    "HuggingFaceModelLoader",
    "CrossProviderLoader",
    
    # Factory functions
    "create_loader",
    "create_cross_provider_loader",
    "get_supported_providers", 
    "get_default_models",
    
    # Prompts (for advanced users)
    "SOLVER_SYSTEM_PROMPT",
    "SOLVER_USER_TEMPLATE",
    "PROOF_GRADER_SYSTEM_PROMPT",
    "CALCULATION_GRADER_SYSTEM_PROMPT",
    "PROOF_GRADER_USER_TEMPLATE",
    "CALCULATION_GRADER_USER_TEMPLATE",
    
    # Configuration constants
    "RESPONSE_FORMAT",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_BASE"
]
