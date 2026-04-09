#!/usr/bin/env python3
"""
Putnam CLI - Simple command-line interface for mathematical problem solving.

This CLI provides easy-to-use commands for testing problems, checking health,
running benchmarks, and managing the system.

Usage:
    putnam solve problem.json                    # Solve a single problem
    putnam test --provider openai                # Quick test
    putnam health                                # Check all providers
    putnam benchmark --quick                     # Quick benchmark
    putnam batch dataset/ --provider anthropic  # Batch evaluation
    
Cross-provider usage:
    putnam solve problem.json --solver-provider kimi --grader-provider openai
    putnam batch dataset/ --solver-provider kimi --grader-provider openai
"""

import asyncio
import json
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, Optional
import os

# Add the loader module to the path
sys.path.append(str(Path(__file__).parent))

from loader import create_loader, create_cross_provider_loader, get_supported_providers, get_default_models


class PutnamCLI:
    """Main CLI class for Putnam problem solver."""
    
    def __init__(self):
        self.verbose = False
    
    def print_banner(self):
        """Print CLI banner."""
        print("🧮 Putnam Mathematical Problem Solver CLI")
        print("=" * 50)
    
    def print_providers(self):
        """Print available providers."""
        print("\n🤖 Available Providers:")
        for provider in get_supported_providers():
            defaults = get_default_models(provider)
            print(f"   • {provider.upper()}")
            print(f"     Solver: {defaults['solver_model']}")
            print(f"     Grader: {defaults['grader_model']}")
        print()
    
    def _create_loader(self, args, loader_kwargs: Optional[Dict] = None) -> Any:
        """
        Create a loader based on command-line arguments.
        Handles both single-provider and cross-provider scenarios.
        
        Args:
            args: Command-line arguments
            loader_kwargs: Additional kwargs for loader creation
            
        Returns:
            ModelLoader instance
        """
        loader_kwargs = loader_kwargs or {}
        
        # Add debug flag if available
        if hasattr(args, 'debug') and args.debug:
            loader_kwargs['debug'] = True
        
        # Handle provider-specific settings
        if hasattr(args, 'vllm_url') and args.vllm_url:
            if args.provider == 'vllm' or (hasattr(args, 'solver_provider') and args.solver_provider == 'vllm'):
                loader_kwargs['solver_kwargs'] = loader_kwargs.get('solver_kwargs', {})
                loader_kwargs['solver_kwargs']['base_url'] = args.vllm_url
            if hasattr(args, 'grader_provider') and args.grader_provider == 'vllm':
                loader_kwargs['grader_kwargs'] = loader_kwargs.get('grader_kwargs', {})
                loader_kwargs['grader_kwargs']['base_url'] = args.vllm_url
        
        if hasattr(args, 'device') and args.device:
            if args.provider == 'huggingface' or (hasattr(args, 'solver_provider') and args.solver_provider == 'huggingface'):
                loader_kwargs['solver_kwargs'] = loader_kwargs.get('solver_kwargs', {})
                loader_kwargs['solver_kwargs']['device'] = args.device
            if hasattr(args, 'grader_provider') and args.grader_provider == 'huggingface':
                loader_kwargs['grader_kwargs'] = loader_kwargs.get('grader_kwargs', {})
                loader_kwargs['grader_kwargs']['device'] = args.device
        
        # Check if we're using cross-provider mode
        if hasattr(args, 'solver_provider') and args.solver_provider:
            # Cross-provider mode
            print(f"🚀 Using solver provider: {args.solver_provider}")
            if hasattr(args, 'grader_provider') and args.grader_provider:
                print(f"🎯 Using grader provider: {args.grader_provider}")
            else:
                print(f"🎯 Using grader provider: {args.solver_provider} (same as solver)")
            
            return create_cross_provider_loader(
                solver_provider=args.solver_provider,
                grader_provider=args.grader_provider if hasattr(args, 'grader_provider') else None,
                solver_model=args.solver_model if hasattr(args, 'solver_model') else None,
                grader_model=args.grader_model if hasattr(args, 'grader_model') else None,
                **loader_kwargs
            )
        else:
            # Single provider mode (backward compatibility)
            provider = args.provider if hasattr(args, 'provider') else "openai"
            print(f"🚀 Using provider: {provider}")
            
            # Handle special cases for single provider
            if provider == 'vllm' and hasattr(args, 'vllm_url'):
                loader_kwargs['base_url'] = args.vllm_url
            elif provider == 'huggingface' and hasattr(args, 'device'):
                loader_kwargs['device'] = args.device
            
            return create_loader(
                provider,
                solver_model=args.solver_model if hasattr(args, 'solver_model') else None,
                grader_model=args.grader_model if hasattr(args, 'grader_model') else None,
                **loader_kwargs
            )
    
    async def cmd_solve(self, args) -> int:
        """Solve a single problem."""
        self.print_banner()
        
        # Setup logging
        import logging
        from datetime import datetime
        from pathlib import Path
        
        # Create log file
        log_dir = Path("solve_logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"solve_debug_{timestamp}.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 Starting solve command, log file: {log_file}")
        
        # Load problem
        try:
            with open(args.problem_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
            logger.info(f"📁 Problem loaded from {args.problem_file}")
        except Exception as e:
            logger.error(f"❌ Error loading problem: {str(e)}")
            return 1
        
        # Setup provider
        loader = self._create_loader(args)
        logger.info(f"🤖 Created loader: solver={loader.solver_model}, grader={loader.grader_model}")
        
        # Health check
        print("🔍 Checking provider health...")
        if not await loader.health_check():
            logger.error("❌ Provider health check failed")
            return 1
        
        # Show problem
        variant_type = args.variant or "original"
        problem_stmt = problem_data.get(variant_type, {}).get('problem_statement', 'N/A')
        logger.info(f"📝 Problem variant: {variant_type}")
        logger.info(f"📄 Problem statement: {problem_stmt[:500]}...")
        
        print(f"\n📝 Problem ({variant_type}):")
        print(f"   {problem_stmt[:200]}{'...' if len(problem_stmt) > 200 else ''}")
        
        # Solve
        print(f"\n⚡ Solving with {loader.solver_model}...")
        logger.info(f"🔄 Starting solve process...")
        
        result = await loader.test_single_problem(
            problem_data, 
            variant_type=variant_type,
            solver_model=args.solver_model,
            grader_model=args.grader_model
        )
        
        # Log detailed results
        logger.info("📊 DETAILED RESULTS:")
        logger.info(f"   Full result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # Analyze solve step
        solve_data = result.get('solve', {})
        solve_status = solve_data.get('status', 'unknown')
        logger.info(f"🔍 SOLVE ANALYSIS:")
        logger.info(f"   Status: {solve_status}")
        
        if solve_status == 'success':
            solution = solve_data.get('solution', 'N/A')
            logger.info(f"   Solution length: {len(solution)} characters")
            logger.info(f"   Solution preview: {solution[:200]}...")
        else:
            error_msg = solve_data.get('error', 'No error message')
            logger.error(f"   Solve error: {error_msg}")
            
        # Analyze grade step
        grade_data = result.get('grade', {})
        grade_status = grade_data.get('status', 'unknown')
        logger.info(f"🔍 GRADE ANALYSIS:")
        logger.info(f"   Status: {grade_status}")
        
        if grade_status == 'success':
            grade = grade_data.get('grade', 'N/A')
            feedback = grade_data.get('detailed_feedback', 'N/A')
            logger.info(f"   Grade: {grade}")
            logger.info(f"   Feedback: {feedback}")
        else:
            error_msg = grade_data.get('error', 'No error message')
            logger.error(f"   Grade error: {error_msg}")
        
        # Show results
        print(f"\n✅ Solution completed!")
        
        # Extract and display grade
        if result.get('grade', {}).get('status') == 'success':
            grade = result.get('grade', {}).get('grade', 'N/A')
            is_correct = result.get('correct', False)
            grade_display = f"{grade} ({'✓' if is_correct else '✗'})"
        else:
            grade_display = 'N/A (grading failed)'
        
        # Extract and display solution
        if result.get('solve', {}).get('status') == 'success':
            solution = result.get('solve', {}).get('solution', 'N/A')
        else:
            solution = 'N/A (solving failed)'
        
        print(f"🎯 Final Grade: {grade_display}")
        print(f"🤖 Solution:")
        print(f"   {solution[:300]}{'...' if len(solution) > 300 else ''}")
        
        if args.verbose:
            grading = result.get('grade', {})
            print(f"\n📊 Grading Details:")
            print(f"   Feedback: {grading.get('detailed_feedback', 'N/A')[:200]}...")
            print(f"   Major Issues: {grading.get('major_issues', 'N/A')}")
            print(f"   Rigor Score: {grading.get('reasoning_rigor_score', 'N/A')}")
        
        # Save detailed results
        results_file = log_dir / f"solve_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Detailed results saved to {results_file}")
        
        # Save if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {args.output}")
        
        print(f"\n📋 Log file created: {log_file}")
        print(f"📋 Results file created: {results_file}")
        
        return 0
            
    async def cmd_test(self, args) -> int:
        """Quick test of a provider."""
        self.print_banner()
        
        # Create simple test problem
        test_problem = {
            'question': 'Calculate 15 + 27.',
            'solution': 'The answer is 42.',
            'problem_type': 'calculation'
        }
        
        try:
            loader = self._create_loader(args)
            
            print("🔍 Health check...")
            if not await loader.health_check():
                print("❌ Health check failed")
                return 1
            
            print("⚡ Running test problem...")
            result = await loader.test_single_problem(test_problem, variant_type='original')
            
            print(f"✅ Test completed!")
            
            # Extract grade information
            if result.get('grade', {}).get('status') == 'success':
                grade = result.get('grade', {}).get('grade', 'N/A')
                is_correct = result.get('correct', False)
                grade_display = f"{grade} ({'✓' if is_correct else '✗'})"
            else:
                grade_display = 'N/A (grading failed)'
            
            # Extract solution
            if result.get('solve', {}).get('status') == 'success':
                solution = result.get('solve', {}).get('solution', 'N/A')
            else:
                solution = 'N/A (solving failed)'
            
            print(f"🎯 Grade: {grade_display}")
            print(f"🤖 Solution: {solution[:100]}...")
            
            return 0
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return 1
    
    async def cmd_health(self, args) -> int:
        """Check health of providers."""
        self.print_banner()
        
        print("🏥 Checking provider health...")
        
        # Import health check
        try:
            from scripts.health_check import HealthChecker
            checker = HealthChecker(detailed=args.detailed)
            
            results = await checker.check_all_providers(args.provider)
            
            # Simple summary
            summary = results['summary']
            print(f"\n📋 Summary: {summary['healthy_providers']}/{summary['total_providers']} providers healthy")
            
            return 0 if summary['healthy_providers'] > 0 else 1
            
        except ImportError:
            print("❌ Health check module not available")
            return 1
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return 1
    
    async def cmd_benchmark(self, args) -> int:
        """Run benchmark."""
        self.print_banner()
        
        print("🏁 Running benchmark...")
        
        try:
            from scripts.benchmark import run_quick_test
            await run_quick_test()
            return 0
            
        except ImportError:
            print("❌ Benchmark module not available")
            return 1
        except Exception as e:
            print(f"❌ Benchmark failed: {str(e)}")
            return 1
    
    async def cmd_batch(self, args) -> int:
        """Run batch evaluation."""
        self.print_banner()
        
        # Handle resume case - simplified version
        if args.resume:
            if not args.resume.exists():
                print(f"❌ Resume checkpoint file not found: {args.resume}")
                return 1
            
            # Simple resume: just read completed problems list
            print(f"📂 Resuming from checkpoint: {args.resume}")
            with open(args.resume) as f:
                checkpoint_data = json.load(f)
            
            # Extract completed problem indices
            completed_indices = checkpoint_data.get('completed_indices', [])
            print(f"   Found {len(completed_indices)} completed problems to skip")
            
            # Still need dataset path for resume
            if not args.dataset_path:
                # Try to get from checkpoint for convenience
                dataset_path = checkpoint_data.get('dataset_path')
                if dataset_path:
                    dataset_path = Path(dataset_path)
                    print(f"   Using dataset path from checkpoint: {dataset_path}")
                else:
                    print("❌ Dataset path is required when resuming")
                    return 1
            else:
                dataset_path = Path(args.dataset_path)
        else:
            # New evaluation
            if not args.dataset_path:
                print("❌ Dataset path is required for new batch evaluation.")
                return 1
            dataset_path = Path(args.dataset_path)
            if not dataset_path.exists():
                print(f"❌ Dataset path not found: {dataset_path}")
                return 1
        
        try:
            # Import batch evaluation functions
            from scripts.batch_evaluate import batch_evaluate, batch_evaluate_cross
            
            # Check if we need to run all variants
            if args.variant == "all" and not args.resume:
                # All available variants
                all_variants = ["original", "descriptive_long", "descriptive_long_confusing",
                               "descriptive_long_misleading", "garbled_string", "kernel_variant"]
                
                print(f"🔄 Running all {len(all_variants)} variants sequentially...")
                
                overall_results = []
                for i, variant in enumerate(all_variants, 1):
                    print(f"\n{'='*60}")
                    print(f"📍 Variant {i}/{len(all_variants)}: {variant}")
                    print(f"{'='*60}")
                    
                    # Determine output file for this variant
                    if args.output:
                        # If output specified, append variant name
                        output_path = Path(args.output)
                        output_file = output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}"
                    else:
                        output_file = None
                    
                    # Run batch evaluation for this variant
                    if hasattr(args, 'solver_provider') and args.solver_provider:
                        # Cross-provider batch evaluation
                        results = await batch_evaluate_cross(
                            dataset_path=dataset_path,
                            solver_provider=args.solver_provider,
                            grader_provider=args.grader_provider if hasattr(args, 'grader_provider') else args.solver_provider,
                            variant_type=variant,
                            max_concurrent=args.concurrent or 3,
                            max_files=args.max_files,
                            solver_model=args.solver_model,
                            grader_model=args.grader_model,
                            output_file=output_file,
                            resume_checkpoint=args.resume,
                            vllm_url=args.vllm_url if hasattr(args, 'vllm_url') else None,
                            device=args.device if hasattr(args, 'device') else None,
                            quick=args.quick if hasattr(args, 'quick') else False
                        )
                    else:
                        # Standard batch evaluation
                        loader_kwargs = {}
                        provider = args.provider or "openai"
                        if provider == 'vllm' and hasattr(args, 'vllm_url'):
                            loader_kwargs['base_url'] = args.vllm_url
                        elif provider == 'huggingface' and hasattr(args, 'device'):
                            loader_kwargs['device'] = args.device
                        
                        # Add quick mode if specified
                        if hasattr(args, 'quick') and args.quick:
                            loader_kwargs['quick'] = True
                            
                        results = await batch_evaluate(
                            dataset_path=dataset_path,
                            provider=provider,
                            variant_type=variant,
                            max_concurrent=args.concurrent or 3,
                            max_files=args.max_files,
                            solver_model=args.solver_model,
                            grader_model=args.grader_model,
                            output_file=output_file,
                            resume_checkpoint=args.resume,
                            **loader_kwargs
                        )
                    
                    print(f"✅ {variant} completed!")
                    print(f"📊 Average grade: {results['summary']['average_grade']:.2f}")
                    print(f"📈 Success rate: {results['summary']['success_rate']:.1f}%")
                    
                    overall_results.append({
                        'variant': variant,
                        'summary': results['summary']
                    })
                    
                    # Wait between variants to ensure clean state
                    if i < len(all_variants):
                        print("\n⏳ Waiting 5 seconds before next variant...")
                        await asyncio.sleep(5)
                
                # Print overall summary
                print(f"\n{'='*60}")
                print("📊 OVERALL SUMMARY")
                print(f"{'='*60}")
                
                for result in overall_results:
                    variant = result['variant']
                    summary = result['summary']
                    print(f"{variant:20s}: Grade {summary['average_grade']:5.2f}, Success {summary['success_rate']:5.1f}%")
                
                return 0
            else:
                # Single variant evaluation
                if hasattr(args, 'solver_provider') and args.solver_provider:
                    # Cross-provider batch evaluation
                    results = await batch_evaluate_cross(
                        dataset_path=dataset_path,
                        solver_provider=args.solver_provider,
                        grader_provider=args.grader_provider if hasattr(args, 'grader_provider') else args.solver_provider,
                        variant_type=args.variant or "original",
                        max_concurrent=args.concurrent or 3,
                        max_files=args.max_files,
                        solver_model=args.solver_model,
                        grader_model=args.grader_model,
                        output_file=Path(args.output) if args.output else None,
                        resume_checkpoint=args.resume,
                        vllm_url=args.vllm_url if hasattr(args, 'vllm_url') else None,
                        device=args.device if hasattr(args, 'device') else None,
                        quick=args.quick if hasattr(args, 'quick') else False
                    )
                else:
                    # Standard batch evaluation
                    loader_kwargs = {}
                    provider = args.provider or "openai"
                    if provider == 'vllm' and hasattr(args, 'vllm_url'):
                        loader_kwargs['base_url'] = args.vllm_url
                    elif provider == 'huggingface' and hasattr(args, 'device'):
                        loader_kwargs['device'] = args.device
                        
                    # Add quick mode if specified
                    if hasattr(args, 'quick') and args.quick:
                        loader_kwargs['quick'] = True
                        
                    results = await batch_evaluate(
                        dataset_path=dataset_path,
                        provider=provider,
                        variant_type=args.variant or "original",
                        max_concurrent=args.concurrent or 3,
                        max_files=args.max_files,
                        solver_model=args.solver_model,
                        grader_model=args.grader_model,
                        output_file=Path(args.output) if args.output else None,
                        resume_checkpoint=args.resume,
                        **loader_kwargs
                    )
                
                print(f"✅ Batch evaluation completed!")
                print(f"📊 Average grade: {results['summary']['average_grade']:.2f}")
                print(f"📈 Success rate: {results['summary']['success_rate']:.1f}%")
                
                return 0
            
        except ImportError:
            print("❌ Batch evaluation module not available")
            return 1
        except Exception as e:
            print(f"❌ Batch evaluation failed: {str(e)}")
            return 1
    
    async def cmd_multi_test(self, args) -> int:
        """Run multi-variant testing."""
        self.print_banner()
        
        provider = args.provider or "openai"
        print(f"🎯 Multi-variant testing with {provider}")
        
        try:
            from scripts.batch_evaluate import batch_evaluate_all_variants
            
            # Run multi-variant evaluation
            results = await batch_evaluate_all_variants(
                dataset_path=Path(args.dataset_path or "dataset"),
                provider=provider,
                variants=args.variants,
                max_concurrent=args.concurrent or 3,
                max_files=args.max_files,
                solver_model=args.solver_model,
                grader_model=args.grader_model,
                output_dir=Path(args.output_dir or "multi_variant_results"),
                base_url=args.vllm_url if provider == 'vllm' else None,
                device=args.device if provider == 'huggingface' else None
            )
            
            print(f"✅ Multi-variant testing completed!")
            metrics = results['aggregate_metrics']
            print(f"📊 Overall average grade: {metrics['overall_average_grade']:.2f}")
            print(f"📈 Overall success rate: {metrics['overall_success_rate']:.1f}%")
            print(f"⏱️ Total time: {results['test_overview']['total_test_time_minutes']:.1f} minutes")
            
            comparison = results['variant_comparison']
            if comparison['best_performing_variant']['variant']:
                print(f"🏆 Best variant: {comparison['best_performing_variant']['variant']} "
                      f"(Grade: {comparison['best_performing_variant']['grade']:.2f})")
            
            return 0
            
        except ImportError:
            print("❌ Multi-variant testing module not available")
            return 1
        except Exception as e:
            print(f"❌ Multi-variant testing failed: {str(e)}")
            return 1

    async def cmd_info(self, args) -> int:
        """Show system information."""
        self.print_banner()
        
        print("ℹ️ System Information")
        print("-" * 30)
        
        # Check environment variables
        print("🔧 Environment Variables:")
        env_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY', 
            'GOOGLE_API_KEY',
            'XAI_API_KEY',
            'MOONSHOT_API_KEY'
        ]
        for var in env_vars:
            value = os.getenv(var)
            status = "✅ Set" if value else "❌ Not set"
            provider = var.replace('_API_KEY', '').replace('MOONSHOT', 'KIMI')
            print(f"   {provider}: {status}")
        
        print()
        self.print_providers()
        
        # Show usage examples
        print("💡 Quick Start Examples:")
        print("   # Single provider:")
        print("   putnam solve dataset/1938-A-1.json")
        print("   putnam test --provider openai")
        print("   putnam batch dataset/ --provider anthropic --max-files 5")
        print("")
        print("   # Cross-provider:")
        print("   putnam solve dataset/1938-A-1.json --solver-provider kimi --grader-provider openai")
        print("   putnam batch dataset/ --solver-provider kimi --grader-provider openai --concurrent 200")
        print("")
        print("   # Full test with all variants:")
        print("   putnam batch dataset/ --variant all --solver-provider kimi --grader-provider openai")
        print("")
        print("   # Resume functionality:")
        print("   putnam batch --resume checkpoint_file.json")
        print("   putnam batch dataset/ --provider openai --resume old_checkpoint_file.json")
        
        return 0


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Putnam Mathematical Problem Solver CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single provider:
    putnam solve problem.json --provider openai
    putnam test --provider anthropic
    putnam batch dataset/ --provider gemini --max-files 10
    
  Cross-provider:
    putnam solve problem.json --solver-provider kimi --grader-provider openai
    putnam batch dataset/ --solver-provider kimi --grader-provider openai --concurrent 200
    putnam batch dataset/ --variant all --solver-provider kimi --grader-provider openai
    
  Resume functionality:
    putnam batch --resume checkpoint_file.json
    putnam batch dataset/ --provider openai --resume old_checkpoint_file.json
        """
    )
    
    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show JSON parsing details)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single problem")
    solve_parser.add_argument("problem_file", type=Path, help="Problem JSON file")
    solve_parser.add_argument("--provider", choices=get_supported_providers(), 
                             help="AI provider (sets both solver and grader)")
    solve_parser.add_argument("--solver-provider", choices=get_supported_providers(),
                             help="Provider for solving")
    solve_parser.add_argument("--grader-provider", choices=get_supported_providers(),

                             help="Provider for grading")
    solve_parser.add_argument("--variant", choices=["original", "descriptive_long", "kernel_variant"], 
                             help="Problem variant")
    solve_parser.add_argument("--solver-model", help="Override solver model")
    solve_parser.add_argument("--grader-model", help="Override grader model")
    solve_parser.add_argument("--output", "-o", type=Path, help="Save results to file")
    solve_parser.add_argument("--debug", action="store_true", help="Enable debug mode (show JSON parsing details)")
    solve_parser.add_argument("--vllm-url", default="http://localhost:8000/v1", 
                             help="VLLM server URL")
    solve_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], 
                             help="Device for HuggingFace")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Quick test of a provider")
    test_parser.add_argument("--provider", choices=get_supported_providers(), 
                            help="AI provider (sets both solver and grader)")
    test_parser.add_argument("--solver-provider", choices=get_supported_providers(),
                            help="Provider for solving")
    test_parser.add_argument("--grader-provider", choices=get_supported_providers(),
                            help="Provider for grading")
    test_parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="VLLM server URL")
    test_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Device for HuggingFace")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check provider health")
    health_parser.add_argument("--provider", choices=get_supported_providers(), 
                              help="Check specific provider only")
    health_parser.add_argument("--detailed", action="store_true", help="Detailed health check")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    benchmark_parser.add_argument("--config", type=Path, help="Configuration file")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch evaluation")
    batch_parser.add_argument("dataset_path", type=Path, nargs='?', help="Dataset directory (required for new runs, optional for resume)")
    batch_parser.add_argument("--provider", choices=get_supported_providers(), 
                         help="AI provider (sets both solver and grader)")
    batch_parser.add_argument("--solver-provider", choices=get_supported_providers(),
                         help="Provider for solving")
    batch_parser.add_argument("--grader-provider", choices=get_supported_providers(),
                         help="Provider for grading")
    batch_parser.add_argument("--variant", choices=["all", "original", "descriptive_long", "descriptive_long_confusing",
                                                "descriptive_long_misleading", "garbled_string", "kernel_variant"],
                         help="Problem variant (use 'all' to run all variants sequentially)")
    batch_parser.add_argument("--max-files", type=int, help="Maximum files to process")
    batch_parser.add_argument("--concurrent", type=int, default=3, help="Concurrent evaluations")
    batch_parser.add_argument("--solver-model", help="Override solver model")
    batch_parser.add_argument("--grader-model", help="Override grader model")
    batch_parser.add_argument("--output", "-o", help="Output file")
    batch_parser.add_argument("--resume", type=Path, help="Resume from checkpoint file")
    batch_parser.add_argument("--debug", action="store_true", help="Enable debug mode (show JSON parsing details)")
    batch_parser.add_argument("--quick", action="store_true", help="Quick mode: allows one retry with 1200s timeout per attempt")
    batch_parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="VLLM server URL")
    batch_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Device for HuggingFace")
    
    # Multi-test command
    multi_parser = subparsers.add_parser("multi-test", help="Run multi-variant testing")
    multi_parser.add_argument("--provider", choices=get_supported_providers(), 
                             help="AI provider (sets both solver and grader)")
    multi_parser.add_argument("--solver-provider", choices=get_supported_providers(),
                             help="Provider for solving")
    multi_parser.add_argument("--grader-provider", choices=get_supported_providers(),
                             help="Provider for grading")
    multi_parser.add_argument("--dataset-path", type=Path, help="Dataset directory path")
    multi_parser.add_argument("--variants", nargs="+",
                             choices=["original", "descriptive_long", "descriptive_long_confusing",
                                     "descriptive_long_misleading", "garbled_string", "kernel_variant"],
                             help="Specific variants to test (default: all)")
    multi_parser.add_argument("--max-files", type=int, help="Maximum files per variant")
    multi_parser.add_argument("--concurrent", type=int, help="Maximum concurrent evaluations")
    multi_parser.add_argument("--solver-model", help="Override solver model")
    multi_parser.add_argument("--grader-model", help="Override grader model")
    multi_parser.add_argument("--output-dir", type=Path, help="Output directory")
    multi_parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="VLLM server URL")
    multi_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Device for HuggingFace")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Create CLI instance
    cli = PutnamCLI()
    cli.verbose = args.verbose
    
    # Route to appropriate command
    try:
        if args.command == "solve":
            return await cli.cmd_solve(args)
        elif args.command == "test":
            return await cli.cmd_test(args)
        elif args.command == "health":
            return await cli.cmd_health(args)
        elif args.command == "benchmark":
            return await cli.cmd_benchmark(args)
        elif args.command == "batch":
            return await cli.cmd_batch(args)
        elif args.command == "multi-test":
            return await cli.cmd_multi_test(args)
        elif args.command == "info":
            return await cli.cmd_info(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏸️ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if cli.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main())) 