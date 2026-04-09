# OpenRouter Integration Guide

## Overview

OpenRouter provides access to multiple AI model providers through a single API endpoint. This integration allows you to use models from OpenAI, Anthropic, Google, Meta, Mistral, and many other providers with a unified interface.

## Setup

### 1. Get an API Key

Sign up at [OpenRouter.ai](https://openrouter.ai) to get your API key.

### 2. Set Environment Variable

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from loader import create_loader

# Create OpenRouter loader with default models
loader = create_loader("openrouter")

# Or specify custom models
loader = create_loader(
    "openrouter",
    solver_model="openai/gpt-4o",
    grader_model="anthropic/claude-3-opus"
)
```

### Direct Instantiation

```python
from loader import OpenRouterModelLoader

loader = OpenRouterModelLoader(
    solver_model="openai/gpt-4o-mini",
    grader_model="openai/gpt-4o",
    api_key="your-api-key",  # Optional, uses env var if not provided
    site_url="https://yoursite.com",  # Optional, for rankings
    site_name="Your Site Name"  # Optional, for rankings
)
```

### Using Cross-Provider Features

One of the key advantages of OpenRouter is the ability to mix models from different providers:

```python
# Use OpenAI for solving and Anthropic for grading
loader = create_loader(
    "openrouter",
    solver_model="openai/gpt-4o",
    grader_model="anthropic/claude-3-opus"
)

# Use Google's Gemini for solving and Meta's Llama for grading
loader = create_loader(
    "openrouter",
    solver_model="google/gemini-pro",
    grader_model="meta-llama/llama-3-70b-instruct"
)
```

## Available Models

OpenRouter supports a wide variety of models. Here are some popular options:

### OpenAI Models
- `openai/gpt-4o` - GPT-4 Optimized
- `openai/gpt-4o-mini` - Smaller, faster GPT-4
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo
- `openai/o1-preview` - O1 Preview (reasoning model)
- `openai/o1-mini` - O1 Mini

### Anthropic Models
- `anthropic/claude-3-opus` - Most capable Claude model
- `anthropic/claude-3-sonnet` - Balanced performance
- `anthropic/claude-3-haiku` - Fastest Claude model
- `anthropic/claude-2.1` - Previous generation
- `anthropic/claude-2` - Previous generation

### Google Models
- `google/gemini-pro` - Gemini Pro
- `google/gemini-pro-vision` - Gemini Pro with vision
- `google/palm-2-codechat-bison` - PaLM 2 for code
- `google/palm-2-chat-bison` - PaLM 2 for chat

### Meta Models
- `meta-llama/llama-3-70b-instruct` - Llama 3 70B
- `meta-llama/llama-3-8b-instruct` - Llama 3 8B
- `meta-llama/codellama-70b-instruct` - CodeLlama 70B

### Mistral Models
- `mistralai/mistral-large` - Mistral Large
- `mistralai/mistral-medium` - Mistral Medium
- `mistralai/mistral-small` - Mistral Small
- `mistralai/mixtral-8x7b-instruct` - Mixtral MoE

### Other Models
- `cohere/command-r-plus` - Cohere Command R+
- `deepseek/deepseek-coder` - DeepSeek Coder
- `qwen/qwen-2-72b-instruct` - Qwen 2 72B

For a complete and up-to-date list, visit: https://openrouter.ai/models

## Testing

Run the test script to verify your setup:

```bash
python test_openrouter.py
```

## Cost Considerations

Different models have different pricing. Generally:
- Mini/small models are cheapest
- Standard models are moderately priced
- Large/opus models are most expensive

Check [OpenRouter pricing](https://openrouter.ai/models) for current rates.

## Troubleshooting

### API Key Issues
- Ensure `OPENROUTER_API_KEY` is set correctly
- Check that your API key has sufficient credits

### Model Availability
- Some models may have limited availability
- Check OpenRouter status page if a model isn't responding

### Rate Limits
- OpenRouter has rate limits that vary by model
- The loader includes automatic retry logic for rate limit errors 