#!/usr/bin/env python3
"""
Health check script for all AI providers.

This script tests connectivity, API keys, and basic functionality for all
supported AI providers. Useful for troubleshooting and verifying setup.

Usage:
    python health_check.py                    # Check all providers
    python health_check.py --provider openai  # Check specific provider
    python health_check.py --detailed         # Detailed diagnostics
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, List, Any
from datetime import datetime
import platform

# Add the loader module to the path
sys.path.append(str(Path(__file__).parent))

from loader import create_loader, get_supported_providers, get_default_models


class HealthChecker:
    """Health checker for AI providers."""
    
    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self.results = {}
        
    async def check_system_info(self) -> Dict[str, Any]:
        """Check system information."""
        import psutil
        
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2),
            'timestamp': datetime.now().isoformat()
        }
    
    async def check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables."""
        env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        }
        
        return {
            var: {
                'set': bool(value),
                'length': len(value) if value else 0,
                'preview': value[:8] + '...' if value and len(value) > 8 else value
            }
            for var, value in env_vars.items()
        }
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check required Python packages."""
        dependencies = {
            'openai': 'OpenAI API client',
            'anthropic': 'Anthropic API client',
            'google-generativeai': 'Google Gemini API client',
            'transformers': 'HuggingFace transformers',
            'torch': 'PyTorch for local models',
            'vllm': 'VLLM for local serving',
            'psutil': 'System monitoring'
        }
        
        results = {}
        for package, description in dependencies.items():
            try:
                if package == 'google-generativeai':
                    import google.generativeai
                    version = getattr(google.generativeai, '__version__', 'unknown')
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                
                results[package] = {
                    'installed': True,
                    'version': version,
                    'description': description
                }
            except ImportError:
                results[package] = {
                    'installed': False,
                    'version': None,
                    'description': description
                }
        
        return results
    
    async def check_provider(self, provider: str) -> Dict[str, Any]:
        """Check a specific AI provider."""
        print(f"🔍 Checking {provider}...")
        
        result = {
            'provider': provider,
            'available': False,
            'health_check_passed': False,
            'error': None,
            'response_time': None,
            'models': {},
            'cost_estimation': None
        }
        
        try:
            # Get default models
            default_models = get_default_models(provider)
            result['models']['defaults'] = default_models
            
            # Provider-specific configuration
            loader_kwargs = {}
            if provider == 'vllm':
                loader_kwargs['base_url'] = 'http://localhost:8000/v1'
            elif provider == 'huggingface':
                loader_kwargs['device'] = 'cpu'  # Use CPU for testing
                # Use smaller models for testing
                loader_kwargs['solver_model'] = 'microsoft/DialoGPT-small'
                loader_kwargs['grader_model'] = 'microsoft/DialoGPT-small'
            
            # Create loader
            start_time = asyncio.get_event_loop().time()
            loader = create_loader(provider, **loader_kwargs)
            creation_time = asyncio.get_event_loop().time() - start_time
            
            result['available'] = True
            result['creation_time'] = creation_time
            
            # Get model info
            model_info = loader.get_model_info()
            result['models']['configured'] = model_info
            
            # Health check
            health_start = asyncio.get_event_loop().time()
            health_passed = await asyncio.wait_for(loader.health_check(), timeout=60)
            health_time = asyncio.get_event_loop().time() - health_start
            
            result['health_check_passed'] = health_passed
            result['response_time'] = health_time
            
            if health_passed:
                # Cost estimation
                try:
                    cost_info = await loader.estimate_cost(10)
                    result['cost_estimation'] = cost_info
                except Exception as e:
                    result['cost_estimation_error'] = str(e)
                
                # Try to list models if available
                if hasattr(loader, 'list_models'):
                    try:
                        available_models = await loader.list_models()
                        result['models']['available'] = available_models[:10]  # Limit output
                    except Exception as e:
                        result['models']['list_error'] = str(e)
            
        except asyncio.TimeoutError:
            result['error'] = 'Health check timed out'
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    async def check_all_providers(self, specific_provider: str = None) -> Dict[str, Any]:
        """Check all providers or a specific one."""
        providers = [specific_provider] if specific_provider else get_supported_providers()
        
        print("🏥 AI Provider Health Check")
        print("=" * 50)
        
        # System information
        if self.detailed:
            print("📊 System Information:")
            system_info = await self.check_system_info()
            for key, value in system_info.items():
                print(f"   {key}: {value}")
            print()
        
        # Environment variables
        print("🔧 Environment Variables:")
        env_info = await self.check_environment_variables()
        for var, info in env_info.items():
            status = "✅" if info['set'] else "❌"
            print(f"   {status} {var}: {'Set' if info['set'] else 'Not set'}")
        print()
        
        # Dependencies
        print("📦 Dependencies:")
        dep_info = await self.check_dependencies()
        for package, info in dep_info.items():
            status = "✅" if info['installed'] else "❌"
            version = f" (v{info['version']})" if info['installed'] and info['version'] != 'unknown' else ""
            print(f"   {status} {package}{version}")
        print()
        
        # Provider checks
        print("🤖 Provider Health Checks:")
        provider_results = {}
        
        for provider in providers:
            provider_result = await self.check_provider(provider)
            provider_results[provider] = provider_result
            
            # Print summary
            if provider_result['available']:
                if provider_result['health_check_passed']:
                    status = "✅"
                    details = f"({provider_result['response_time']:.2f}s)"
                else:
                    status = "⚠️"
                    details = "(Health check failed)"
            else:
                status = "❌"
                details = f"({provider_result['error']})"
            
            print(f"   {status} {provider.upper()}: {details}")
        
        print()
        
        # Summary
        total_providers = len(providers)
        healthy_providers = sum(1 for r in provider_results.values() 
                              if r['available'] and r['health_check_passed'])
        
        print("📋 Summary:")
        print(f"   Total providers checked: {total_providers}")
        print(f"   Healthy providers: {healthy_providers}")
        print(f"   Success rate: {healthy_providers/total_providers*100:.1f}%")
        
        # Detailed results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_providers': total_providers,
                'healthy_providers': healthy_providers,
                'success_rate': healthy_providers/total_providers*100
            },
            'environment': env_info,
            'dependencies': dep_info,
            'providers': provider_results
        }
        
        if self.detailed:
            final_results['system'] = system_info
        
        return final_results
    
    async def run_diagnostics(self, provider: str) -> Dict[str, Any]:
        """Run detailed diagnostics for a specific provider."""
        print(f"🔧 Running detailed diagnostics for {provider}...")
        
        result = await self.check_provider(provider)
        
        # Additional detailed checks
        if result['available'] and result['health_check_passed']:
            print(f"✅ {provider} is healthy!")
            
            # Test with a simple problem
            print("🧪 Testing with a simple math problem...")
            try:
                loader_kwargs = {}
                if provider == 'vllm':
                    loader_kwargs['base_url'] = 'http://localhost:8000/v1'
                elif provider == 'huggingface':
                    loader_kwargs['device'] = 'cpu'
                    loader_kwargs['solver_model'] = 'microsoft/DialoGPT-small'
                    loader_kwargs['grader_model'] = 'microsoft/DialoGPT-small'
                
                loader = create_loader(provider, **loader_kwargs)
                
                # Simple test problem
                test_problem = {
                    'original': {
                        'problem_statement': 'What is 2 + 2?',
                        'solution': 'The answer is 4.',
                        'problem_type': 'calculation'
                    }
                }
                
                start_time = asyncio.get_event_loop().time()
                test_result = await asyncio.wait_for(
                    loader.test_single_problem(test_problem, variant_type='original'),
                    timeout=120
                )
                test_time = asyncio.get_event_loop().time() - start_time
                
                result['test_problem'] = {
                    'success': True,
                    'time': test_time,
                    'grade': 10 if test_result.get('correct', False) else 0,
                    'solution_length': len(test_result.get('solve', {}).get('solution', ''))
                }
                print(f"   ✅ Test completed in {test_time:.2f}s")
                print(f"   📊 Grade: {10 if test_result.get('correct', False) else 0} ({'CORRECT' if test_result.get('correct', False) else 'INCORRECT'})")
                
            except asyncio.TimeoutError:
                result['test_problem'] = {'success': False, 'error': 'Test timed out'}
                print("   ⚠️ Test problem timed out")
            except Exception as e:
                result['test_problem'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Test problem failed: {str(e)}")
        
        return result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Health check for AI providers")
    parser.add_argument("--provider", choices=get_supported_providers(),
                       help="Check specific provider only")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed system information")
    parser.add_argument("--diagnostics", action="store_true",
                       help="Run detailed diagnostics (requires --provider)")
    parser.add_argument("--output", type=Path,
                       help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output, save to file only")
    
    args = parser.parse_args()
    
    if args.diagnostics and not args.provider:
        print("❌ Error: --diagnostics requires --provider")
        return 1
    
    # Redirect output if quiet
    if args.quiet:
        import io
        sys.stdout = io.StringIO()
    
    checker = HealthChecker(detailed=args.detailed)
    
    try:
        if args.diagnostics:
            results = await checker.run_diagnostics(args.provider)
        else:
            results = await checker.check_all_providers(args.provider)
        
        # Save to file if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            if not args.quiet:
                print(f"\n💾 Results saved to {args.output}")
        
        # Print JSON if quiet mode
        if args.quiet:
            sys.stdout = sys.__stdout__
            print(json.dumps(results, indent=2))
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ Health check interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Health check failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main())) 