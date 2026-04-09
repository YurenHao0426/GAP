# Putnam Mathematical Problem Solver

A comprehensive system for evaluating AI models on mathematical problem solving using the Putnam Competition dataset. This project provides a unified interface for testing multiple AI providers (cloud and local) on complex mathematical problems with automated grading.

## Features

- **Multi-Provider Support**: 8 AI providers including OpenAI, Anthropic, Google Gemini, xAI, Kimi, VLLM, VLLM Direct, and HuggingFace
- **Dynamic Model Selection**: Runtime model configuration for optimal cost/performance
- **Robust Evaluation**: Specialized prompts for mathematical problem solving and grading
- **Local Model Support**: Run models locally via VLLM server or direct HuggingFace inference
- **GPU Optimization**: Tested on RTX 3060 with CUDA support
- **Cost Estimation**: Built-in cost calculation for different providers
- **Async Processing**: Efficient handling of large datasets
- **Error Recovery**: Intelligent retry logic and JSON parsing fallbacks
- **Unified CLI**: Single command-line interface for all operations
- **Progress Tracking**: Real-time progress bars with success/failure statistics
- **Multi-Variant Testing**: Test all 6 problem variants with a single command

## Architecture

The project follows a clean, modular architecture:

```
putnam-bench-anon/
├── putnam_cli.py          # 🎯 Main CLI interface (your primary entry point)
├── loader/                # 🔧 Core evaluation engine
│   ├── __init__.py
│   ├── providers/         # AI provider implementations
│   └── utils/             # Utility functions
├── scripts/               # 📋 Internal scripts (used by CLI)
│   ├── batch_evaluate.py  # Batch evaluation logic
│   ├── health_check.py    # Health checking utilities
│   ├── benchmark.py       # Performance benchmarking
│   └── compare_*.py       # Analysis scripts
├── dataset/               # 📚 Problem dataset
├── results/               # 📊 Evaluation results
└── requirements*.txt      # 📦 Dependency management
```

**Key principle**: Use `putnam_cli.py` for all operations - it provides a clean, consistent interface to all functionality.

## Installation

### Quick Start (Automated)

The easiest way to get started is using our automated installation script:

```bash
# Clone the repository
git clone <repository-url>
cd putnam-bench-anon

# One-command setup (installs everything and configures)
python install.py

# Or quick install (core packages only)
python install.py --quick
```

### Manual Installation

#### Environment Setup

We recommend using a dedicated conda environment to avoid dependency conflicts:

```bash
# Create dedicated environment
conda create -n putnam-local python=3.10 -y
conda activate putnam-local

# Clone the repository
git clone <repository-url>
cd putnam-bench-anon
```

#### Choose Your Installation Type

**Option 1: Full Installation (recommended)**
```bash
# Install all dependencies including local model support
pip install -r requirements.txt

# Install PyTorch with CUDA support (for local models)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Quick tqdm install for progress bars:**
```bash
pip install tqdm
```

**Option 2: Minimal Installation (cloud providers only)**
```bash
# Install only core dependencies for cloud providers
pip install -r requirements-minimal.txt
```

**Option 3: Local Models Only**
```bash
# Install dependencies for local model inference
pip install -r requirements-local.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Option 4: Custom Installation**
```bash
# Install core packages manually
pip install openai anthropic google-generativeai transformers accelerate six

# Optional: For local models
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Optional: For VLLM (see performance notes below)
pip install vllm
```

### GPU Requirements

- **Recommended**: RTX 3060+ with 6GB+ VRAM
- **Minimum**: Any CUDA-compatible GPU with 4GB+ VRAM
- **CPU fallback**: Supported but not recommended for performance

### Requirements Files Explained

The project includes several requirements files for different use cases:

- **`requirements.txt`**: Complete installation with all features (recommended)
  - Includes cloud providers, local models, data analysis tools
  - Best for full functionality and development

- **`requirements-minimal.txt`**: Minimal installation for cloud providers only
  - OpenAI, Anthropic, Google Gemini support
  - Smallest footprint, fastest installation
  - Best for cloud-only usage

- **`requirements-local.txt`**: Specialized for local model inference
  - HuggingFace transformers, VLLM, GPU utilities
  - Includes model optimization tools
  - Best for privacy-focused or offline usage

Choose the requirements file that matches your intended usage pattern.

## Quick Start

### Basic Usage

```python
import asyncio
from loader import create_loader
import json

async def main():
    # Create a loader for any provider
    loader = create_loader("openai", solver_model="gpt-4o-mini", grader_model="gpt-4o-mini")
    
    # Load a problem
    with open("dataset/2000-A-1.json") as f:
        problem = json.load(f)
    
    # Test the problem
    result = await loader.test_single_problem(problem, variant_type="original")
    print(f"Grade: {result['final_grade']}")
    print(f"Solution: {result['solution']}")

asyncio.run(main())
```

### Command Line Usage

```bash
# Activate environment first
conda activate putnam-local

# Use the unified CLI for all operations
python putnam_cli.py info                    # Show system info
python putnam_cli.py health --provider openai
python putnam_cli.py test --provider openai
python putnam_cli.py solve dataset/2000-A-1.json --provider openai
python putnam_cli.py batch dataset/ --provider openai --max-files 10
python putnam_cli.py multi-test --provider openai --max-files 50 --concurrent 100
```

## Multi-Provider Support

The system supports **8 AI providers** with a unified interface:

### Cloud Providers

#### 1. **OpenAI**
- **Models**: GPT-4o, GPT-4o-mini, o1, o3
- **Best for**: High-quality solutions and grading
- **Setup**: Requires `OPENAI_API_KEY` environment variable

#### 2. **Anthropic**
- **Models**: Claude-3.5-Sonnet, Claude-3.5-Haiku
- **Best for**: Detailed reasoning and explanation
- **Setup**: Requires `ANTHROPIC_API_KEY` environment variable

#### 3. **Google Gemini**
- **Models**: Gemini-1.5-Pro, Gemini-1.5-Flash
- **Best for**: Cost-effective high-performance solving
- **Setup**: Requires `GOOGLE_API_KEY` environment variable

#### 4. **xAI**
- **Models**: Grok-3, Grok-2
- **Best for**: Advanced reasoning and mathematical problem solving
- **Setup**: Requires `XAI_API_KEY` environment variable

#### 5. **Kimi (Moonshot AI)**
- **Models**: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
- **Best for**: Chinese and English mathematical problem solving
- **Setup**: Requires `MOONSHOT_API_KEY` environment variable

### Local Providers

#### 6. **HuggingFace**
- **Models**: Any HuggingFace model (tested: GPT-2, DialoGPT)
- **Performance**: Fast loading (~40s first time, then cached)
- **Cost**: Free after setup
- **Privacy**: Complete local inference
- **Best for**: Development, testing, cost-sensitive applications

#### 7. **VLLM Server**
- **Models**: Any VLLM-compatible model
- **Best for**: Multi-user server deployment
- **Setup**: Requires running separate VLLM server
- **Performance**: Good for sustained workloads

#### 8. **VLLM Direct** ⚠️
- **Models**: Any VLLM-compatible model
- **Performance**: **NOT RECOMMENDED** - 72+ second initialization
- **Issue**: Extremely slow first load due to graph compilation
- **Use case**: Only for research/benchmarking VLLM internals

```python
# NOT RECOMMENDED for interactive use
loader = create_loader("vllm_direct", solver_model="gpt2")  # Takes 72+ seconds!
```

## Performance Test Results

Based on testing with RTX 3060 Laptop GPU (6GB VRAM):

| Provider | First Load | Subsequent | GPU Memory | Recommendation |
|----------|------------|-----------|------------|----------------|
| API Clients | 1-2s | 1-2s | 0GB (cloud) | ⭐ **Best** |
| HuggingFace | ~40s | <1s | 0.27GB | ⭐ **Excellent** |
| VLLM Server | ~30s | <1s | Variable | ✅ Good |
| VLLM Direct | **72+s** | <1s | 0.24GB | ❌ **Avoid** |

## Configuration Examples

### HuggingFace Local Setup

```python
# GPU inference (recommended)
loader = create_loader(
    "huggingface",
    solver_model="gpt2",
    grader_model="gpt2",
    device="cuda"
)

# CPU inference (slower)
loader = create_loader(
    "huggingface",
    solver_model="gpt2",
    grader_model="gpt2",
    device="cpu"
)
```

### OpenAI Configuration

```python
# High-quality setup
loader = create_loader(
    "openai",
    solver_model="gpt-4o-mini",
    grader_model="o3"
)
```

### Kimi Configuration

```python
# Standard 8k context window
loader = create_loader(
    "kimi",
    solver_model="moonshot-v1-8k",
    grader_model="moonshot-v1-8k"
)

# Large context window for complex problems
loader = create_loader(
    "kimi",
    solver_model="moonshot-v1-128k",
    grader_model="moonshot-v1-128k"
)
```

### VLLM Server Setup

```bash
# Start VLLM server first (separate terminal):
conda activate putnam-local
vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8000
```

```python
# Then use the client
loader = create_loader(
    "vllm",
    solver_model="meta-llama/Llama-3.2-8B-Instruct",
    base_url="http://localhost:8000/v1"
)
```

## Recommended Usage Patterns

### For Development/Testing
```python
# Fast iteration with local models
loader = create_loader("huggingface", solver_model="gpt2", device="cuda")
```

### For Production/Important Results
```python
# High-quality cloud models
loader = create_loader("openai", solver_model="gpt-4o-mini", grader_model="o3")
loader = create_loader("anthropic", solver_model="claude-3-5-sonnet")
```

### For Privacy-Sensitive Work
```python
# Completely local inference
loader = create_loader("huggingface", solver_model="gpt2", device="cuda")
```

### For Cost Optimization
```python
# Free local models after setup
loader = create_loader("huggingface", solver_model="gpt2", device="cuda")

# Or cost-effective cloud
loader = create_loader("openai", solver_model="gpt-4o-mini")
```

## Dataset Structure

The dataset contains Putnam Competition problems in JSON format:

```json
{
    "original": {
        "problem_statement": "Mathematical problem...",
        "solution": "Step-by-step solution...",
        "problem_type": "proof"
    },
    "descriptive_long": {...},
    "kernel_variant": {...}
}
```

### Problem Variants

- **original**: The standard problem statement
- **descriptive_long**: More verbose problem description
- **descriptive_long_confusing**: Intentionally confusing wording
- **descriptive_long_misleading**: Misleading formulation
- **garbled_string**: Corrupted text version
- **kernel_variant**: Minimal essential formulation

## Advanced Usage

### GPU Memory Management

```python
# Check GPU status
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Cost Estimation

```python
# Estimate costs before processing
cost_info = await loader.estimate_cost(
    num_problems=100,
    avg_problem_length=1000,
    avg_solution_length=2000
)

print(f"Estimated cost: ${cost_info['total_cost']:.2f}")
print(f"Cost per problem: ${cost_info['cost_per_problem']:.4f}")
```

### Batch Processing

```python
async def process_dataset(dataset_path, provider="openai"):
    loader = create_loader(provider)
    
    problems = []
    for file_path in Path(dataset_path).glob("*.json"):
        with open(file_path) as f:
            problems.append(json.load(f))
    
    # Process problems concurrently
    tasks = [
        loader.test_single_problem(problem, variant_type="original")
        for problem in problems
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

#### Incremental Saving and Resume Support

The batch evaluation now supports incremental saving and resume functionality:

```bash
# Start a new evaluation with incremental saving
python putnam_cli.py batch dataset/ --provider openai --output results/my_results.json

# If interrupted, resume from checkpoint (no other arguments needed!)
python putnam_cli.py batch --resume results/checkpoint_my_results_20240101_120000.json

# The checkpoint file is automatically created and updated after each problem
# It contains all progress and can be used to resume if the process is interrupted
```

**Features:**
- **Automatic Checkpointing**: Results are saved after each problem completes
- **Resume Support**: Continue from where you left off if interrupted
- **Backward Compatibility**: Existing checkpoint files continue to work
- **Atomic Saves**: Uses temporary files to ensure data integrity
- **Progress Tracking**: Shows saved/completed problems in real-time
- **Automatic Cleanup**: Checkpoint files are removed after successful completion

**Resume Usage:**
```bash
# For new checkpoint files (created after this update)
python putnam_cli.py batch --resume checkpoint_file.json

# For existing checkpoint files (created before this update)
python putnam_cli.py batch dataset/ --provider openai --resume old_checkpoint_file.json
```

The system automatically detects checkpoint format and handles both old and new formats seamlessly.

### Multi-Variant Testing

Test all 6 problem variants with a single command using the unified CLI:

```bash
# Test all variants with OpenAI (50 files per variant)
python putnam_cli.py multi-test --provider openai --max-files 50

# Test specific variants only
python putnam_cli.py multi-test --provider anthropic --variants original kernel_variant --max-files 25

# Test with custom models and high concurrency
python putnam_cli.py multi-test --provider openai --solver-model gpt-4.1-nano --grader-model o3 --max-files 1051 --concurrent 1100

# Test with local models
python putnam_cli.py multi-test --provider huggingface --device cuda --max-files 100

# Test with VLLM server
python putnam_cli.py multi-test --provider vllm --vllm-url http://localhost:8000/v1 --max-files 50
```

**Available Problem Variants:**
- `original` - Standard mathematical problems
- `descriptive_long` - Clear variable renaming
- `descriptive_long_confusing` - Random unrelated words (marshmallow, armadillo, etc.)
- `descriptive_long_misleading` - Misleading variable names (nonpositiveterm, negativeinitial, etc.)
- `garbled_string` - Completely scrambled variable names  
- `kernel_variant` - Simplified core mathematical version

**Output Structure:**
```
multi_variant_results/
├── openai_gpt-4o-mini_20241201_143022/
│   ├── original_20241201_143022.json
│   ├── descriptive_long_20241201_143022.json
│   ├── ...
│   └── SUMMARY_openai_gpt-4o-mini_20241201_143022.json
└── multi_config_comparison_20241201_143022.json
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA compatibility
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA issues, reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Memory Issues
```python
# Clear GPU cache
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### VLLM Performance Issues
- **Problem**: VLLM Direct takes 72+ seconds to load
- **Solution**: Use HuggingFace local or VLLM server instead
- **Alternative**: Use cloud providers for better speed

## Testing

```bash
# Activate environment
conda activate putnam-local

# Quick health checks
python -c "
import asyncio
from loader import create_loader

async def test():
    # Test available providers
    providers = [
        ('openai', 'gpt-4o-mini'), 
        ('huggingface', 'gpt2'),
        ('anthropic', 'claude-3-5-haiku'),
        ('kimi', 'moonshot-v1-8k')
    ]
    for provider, model in providers:
        try:
            loader = create_loader(provider, solver_model=model)
            result = await loader.health_check()
            print(f'{provider}: {"✅ Ready" if result else "❌ Failed"}')
        except Exception as e:
            print(f'{provider}: ❌ Error - {e}')

asyncio.run(test())
"
```

## Performance Recommendations

### For Speed ⚡
1. **API Clients**: Sub-2s response (OpenAI, Anthropic, xAI, Gemini)
2. **HuggingFace**: Fast after first load, completely local
3. **VLLM Server**: Good for sustained workloads

### For Cost 💰
1. **HuggingFace**: Free after setup, local inference
2. **OpenAI GPT-4o-mini**: Cost-effective cloud option
3. **Gemini Flash**: Good price/performance ratio

### For Privacy 🔒
1. **HuggingFace**: Complete local inference
2. **VLLM Server**: Local deployment with server architecture

### For Quality 🎯
1. **OpenAI o3**: Excellent grading capability
2. **Claude-3.5-Sonnet**: Detailed explanations
3. **Gemini Pro**: Strong mathematical reasoning

## API Reference

### Supported Providers
- `openai`: OpenAI GPT models
- `anthropic`: Anthropic Claude models  
- `gemini`: Google Gemini models
- `xai`: xAI Grok models
- `kimi`: Kimi/Moonshot models
- `vllm`: VLLM server models
- `vllm_direct`: Direct VLLM API (slow)
- `huggingface`: HuggingFace transformers

### Factory Function

```python
create_loader(provider: str, **kwargs) -> ModelLoader
```

### Key Methods

```python
# Test a single problem
await loader.test_single_problem(problem_data, variant_type)

# Health check
await loader.health_check()

# Cost estimation
await loader.estimate_cost(num_problems)

# Model information
loader.get_model_info()
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test with the putnam-local environment
4. Add tests for new functionality
5. Submit a pull request

## License

[License information]

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{putnam-bench-anon,
    title={Putnam Mathematical Problem Solver},
    year={2024},
    howpublished={\url{<repository-url>}}
}
```