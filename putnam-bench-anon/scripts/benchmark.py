#!/usr/bin/env python3
"""
Benchmark script for comparing AI providers and models on mathematical problems.

This script runs comparative evaluations across multiple providers, models, and
problem variants to assess performance, accuracy, cost, and speed trade-offs.

Usage:
    python benchmark.py --config benchmark_config.json
    python benchmark.py --quick-test  # Quick 3-problem test across all providers
    python benchmark.py --providers openai anthropic --models gpt-4o-mini claude-3-5-haiku
"""

import asyncio
import json
import sys
import time
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import itertools
import statistics

# Add the loader module to the path
sys.path.append(str(Path(__file__).parent))

from loader import create_loader, get_supported_providers, get_default_models


class BenchmarkRunner:
    """Benchmark runner for AI providers."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def load_test_problems(self, dataset_path: Path, max_problems: int = 10) -> List[Dict[str, Any]]:
        """Load test problems from dataset."""
        json_files = list(dataset_path.glob("*.json"))[:max_problems]
        
        problems = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file.name)
                    problems.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {str(e)}")
        
        return problems
    
    async def run_single_configuration(self, 
                                     provider: str, 
                                     solver_model: str, 
                                     grader_model: str,
                                     problems: List[Dict[str, Any]], 
                                     variant_type: str = "original",
                                     **loader_kwargs) -> Dict[str, Any]:
        """Run benchmark for a single provider/model configuration."""
        config_name = f"{provider}_{solver_model}_{grader_model}".replace("/", "_").replace("-", "_")
        self.logger.info(f"🚀 Testing configuration: {config_name}")
        
        result = {
            'configuration': {
                'provider': provider,
                'solver_model': solver_model,
                'grader_model': grader_model,
                'variant_type': variant_type,
                'loader_kwargs': loader_kwargs
            },
            'metrics': {},
            'problems': [],
            'errors': []
        }
        
        try:
            # Create loader
            loader = create_loader(
                provider, 
                solver_model=solver_model, 
                grader_model=grader_model,
                **loader_kwargs
            )
            
            # Health check
            if not await loader.health_check():
                raise RuntimeError(f"Health check failed for {provider}")
            
            # Cost estimation
            cost_info = await loader.estimate_cost(len(problems))
            result['metrics']['estimated_cost'] = cost_info
            
            # Process each problem
            start_time = time.time()
            grades = []
            processing_times = []
            
            for i, problem in enumerate(problems):
                problem_start = time.time()
                
                try:
                    problem_result = await loader.test_single_problem(
                        problem, 
                        variant_type=variant_type
                    )
                    
                    processing_time = time.time() - problem_start
                    # Convert boolean 'correct' to numeric grade (10 for correct, 0 for incorrect)
                    grade = 10 if problem_result.get('correct', False) else 0
                    
                    grades.append(grade)
                    processing_times.append(processing_time)
                    
                    result['problems'].append({
                        'source_file': problem.get('_source_file', f'problem_{i}'),
                        'grade': grade,
                        'processing_time': processing_time,
                        'solution_length': len(problem_result.get('solution', '')),
                        'grading_feedback_length': len(str(problem_result.get('grading_result', {}).get('feedback', '')))
                    })
                    
                    self.logger.info(f"   Problem {i+1}/{len(problems)}: Grade {grade} ({processing_time:.2f}s)")
                    
                except Exception as e:
                    error_info = {
                        'problem_index': i,
                        'source_file': problem.get('_source_file', f'problem_{i}'),
                        'error': str(e),
                        'processing_time': time.time() - problem_start
                    }
                    result['errors'].append(error_info)
                    self.logger.error(f"   Problem {i+1}/{len(problems)} failed: {str(e)}")
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            if grades:
                result['metrics'].update({
                    'total_problems': len(problems),
                    'successful_problems': len(grades),
                    'failed_problems': len(result['errors']),
                    'success_rate': len(grades) / len(problems) * 100,
                    'average_grade': statistics.mean(grades),
                    'median_grade': statistics.median(grades),
                    'grade_std': statistics.stdev(grades) if len(grades) > 1 else 0,
                    'max_grade': max(grades),
                    'min_grade': min(grades),
                    'total_time': total_time,
                    'average_time_per_problem': statistics.mean(processing_times),
                    'median_time_per_problem': statistics.median(processing_times),
                    'total_time_successful': sum(processing_times),
                    'throughput_problems_per_minute': len(grades) / (total_time / 60) if total_time > 0 else 0
                })
            else:
                result['metrics'].update({
                    'total_problems': len(problems),
                    'successful_problems': 0,
                    'failed_problems': len(result['errors']),
                    'success_rate': 0,
                    'total_time': total_time,
                    'error_rate': 100
                })
            
            self.logger.info(f"✅ Configuration completed: {result['metrics']['success_rate']:.1f}% success, "
                           f"avg grade: {result['metrics'].get('average_grade', 0):.2f}")
            
        except Exception as e:
            result['metrics']['fatal_error'] = str(e)
            self.logger.error(f"❌ Configuration failed: {str(e)}")
        
        return result
    
    async def run_comparative_benchmark(self, 
                                      configurations: List[Dict[str, Any]], 
                                      problems: List[Dict[str, Any]],
                                      variant_type: str = "original") -> Dict[str, Any]:
        """Run comparative benchmark across multiple configurations."""
        self.logger.info(f"🏁 Starting comparative benchmark with {len(configurations)} configurations")
        self.logger.info(f"📊 Testing {len(problems)} problems with variant: {variant_type}")
        
        benchmark_start = time.time()
        results = []
        
        for i, config in enumerate(configurations):
            self.logger.info(f"\n📋 Configuration {i+1}/{len(configurations)}")
            
            provider = config['provider']
            solver_model = config.get('solver_model')
            grader_model = config.get('grader_model')
            loader_kwargs = config.get('loader_kwargs', {})
            
            # Use defaults if not specified
            if not solver_model or not grader_model:
                defaults = get_default_models(provider)
                solver_model = solver_model or defaults['solver_model']
                grader_model = grader_model or defaults['grader_model']
            
            config_result = await self.run_single_configuration(
                provider=provider,
                solver_model=solver_model,
                grader_model=grader_model,
                problems=problems,
                variant_type=variant_type,
                **loader_kwargs
            )
            
            results.append(config_result)
        
        total_benchmark_time = time.time() - benchmark_start
        
        # Generate comparison report
        report = self.generate_comparison_report(results, total_benchmark_time)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_file = self.output_dir / f"benchmark_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'benchmark_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_configurations': len(configurations),
                    'total_problems': len(problems),
                    'variant_type': variant_type,
                    'total_time': total_benchmark_time
                },
                'configurations': configurations,
                'results': results,
                'comparison_report': report
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Detailed results saved to {detailed_file}")
        
        return report
    
    def generate_comparison_report(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Generate comparison report from benchmark results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 BENCHMARK COMPARISON REPORT")
        self.logger.info("="*60)
        
        # Filter successful results
        successful_results = [r for r in results if r['metrics'].get('success_rate', 0) > 0]
        
        if not successful_results:
            self.logger.warning("⚠️ No successful configurations found!")
            return {'error': 'No successful configurations'}
        
        # Ranking by different metrics
        rankings = {
            'accuracy': sorted(successful_results, key=lambda x: x['metrics']['average_grade'], reverse=True),
            'speed': sorted(successful_results, key=lambda x: x['metrics']['average_time_per_problem']),
            'throughput': sorted(successful_results, key=lambda x: x['metrics']['throughput_problems_per_minute'], reverse=True),
            'success_rate': sorted(successful_results, key=lambda x: x['metrics']['success_rate'], reverse=True)
        }
        
        # Print rankings
        for metric, ranked_results in rankings.items():
            self.logger.info(f"\n🏆 Top 3 by {metric.upper()}:")
            for i, result in enumerate(ranked_results[:3]):
                config = result['configuration']
                metrics = result['metrics']
                provider = config['provider']
                solver = config['solver_model']
                
                if metric == 'accuracy':
                    value = f"{metrics['average_grade']:.2f}"
                elif metric == 'speed':
                    value = f"{metrics['average_time_per_problem']:.2f}s"
                elif metric == 'throughput':
                    value = f"{metrics['throughput_problems_per_minute']:.1f} prob/min"
                elif metric == 'success_rate':
                    value = f"{metrics['success_rate']:.1f}%"
                
                self.logger.info(f"   {i+1}. {provider}/{solver}: {value}")
        
        # Calculate cost efficiency
        cost_efficiency = []
        for result in successful_results:
            metrics = result['metrics']
            cost_info = metrics.get('estimated_cost', {})
            total_cost = cost_info.get('total_cost', 0)
            avg_grade = metrics.get('average_grade', 0)
            
            if total_cost > 0 and avg_grade > 0:
                efficiency = avg_grade / total_cost  # Grade per unit cost
                cost_efficiency.append({
                    'result': result,
                    'efficiency': efficiency,
                    'cost': total_cost,
                    'grade': avg_grade
                })
        
        if cost_efficiency:
            cost_efficiency.sort(key=lambda x: x['efficiency'], reverse=True)
            self.logger.info(f"\n💰 Top 3 by COST EFFICIENCY (Grade/Cost):")
            for i, item in enumerate(cost_efficiency[:3]):
                config = item['result']['configuration']
                provider = config['provider']
                solver = config['solver_model']
                self.logger.info(f"   {i+1}. {provider}/{solver}: {item['efficiency']:.2f} "
                               f"(Grade: {item['grade']:.2f}, Cost: {item['cost']:.4f})")
        
        # Overall statistics
        all_grades = []
        all_times = []
        all_success_rates = []
        
        for result in successful_results:
            metrics = result['metrics']
            all_grades.append(metrics['average_grade'])
            all_times.append(metrics['average_time_per_problem'])
            all_success_rates.append(metrics['success_rate'])
        
        self.logger.info(f"\n📈 OVERALL STATISTICS:")
        self.logger.info(f"   Configurations tested: {len(results)}")
        self.logger.info(f"   Successful configurations: {len(successful_results)}")
        self.logger.info(f"   Average grade across all: {statistics.mean(all_grades):.2f}")
        self.logger.info(f"   Average time per problem: {statistics.mean(all_times):.2f}s")
        self.logger.info(f"   Average success rate: {statistics.mean(all_success_rates):.1f}%")
        self.logger.info(f"   Total benchmark time: {total_time/60:.2f} minutes")
        
        # Generate final report
        report = {
            'summary': {
                'total_configurations': len(results),
                'successful_configurations': len(successful_results),
                'overall_avg_grade': statistics.mean(all_grades) if all_grades else 0,
                'overall_avg_time': statistics.mean(all_times) if all_times else 0,
                'overall_avg_success_rate': statistics.mean(all_success_rates) if all_success_rates else 0,
                'total_benchmark_time': total_time
            },
            'rankings': {
                metric: [
                    {
                        'provider': r['configuration']['provider'],
                        'solver_model': r['configuration']['solver_model'],
                        'grader_model': r['configuration']['grader_model'],
                        'score': r['metrics'][metric_key]
                    }
                    for r in ranked[:5]  # Top 5
                ] for metric, ranked in rankings.items() 
                for metric_key in [{'accuracy': 'average_grade', 'speed': 'average_time_per_problem', 
                                 'throughput': 'throughput_problems_per_minute', 'success_rate': 'success_rate'}[metric]]
            },
            'cost_efficiency': [
                {
                    'provider': item['result']['configuration']['provider'],
                    'solver_model': item['result']['configuration']['solver_model'],
                    'efficiency': item['efficiency'],
                    'grade': item['grade'],
                    'cost': item['cost']
                }
                for item in cost_efficiency[:5]
            ] if cost_efficiency else []
        }
        
        return report


async def run_quick_test():
    """Run a quick test across all providers with 3 problems."""
    runner = BenchmarkRunner()
    
    # Load 3 test problems
    problems = await runner.load_test_problems(Path("dataset"), max_problems=3)
    if not problems:
        print("❌ No test problems found in dataset directory")
        return
    
    # Default configurations for all providers
    configurations = []
    for provider in get_supported_providers():
        config = {'provider': provider}
        
        # Provider-specific settings
        if provider == 'vllm':
            config['loader_kwargs'] = {'base_url': 'http://localhost:8000/v1'}
        elif provider == 'huggingface':
            config['loader_kwargs'] = {
                'device': 'cpu',
                'solver_model': 'microsoft/DialoGPT-small',
                'grader_model': 'microsoft/DialoGPT-small'
            }
        
        configurations.append(config)
    
    # Run benchmark
    await runner.run_comparative_benchmark(configurations, problems)


async def run_custom_benchmark(config_file: Path):
    """Run benchmark from configuration file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    runner = BenchmarkRunner(Path(config.get('output_dir', 'benchmark_results')))
    
    # Load problems
    dataset_path = Path(config.get('dataset_path', 'dataset'))
    max_problems = config.get('max_problems', 10)
    variant_type = config.get('variant_type', 'original')
    
    problems = await runner.load_test_problems(dataset_path, max_problems)
    if not problems:
        print(f"❌ No problems found in {dataset_path}")
        return
    
    # Load configurations
    configurations = config.get('configurations', [])
    if not configurations:
        print("❌ No configurations specified in config file")
        return
    
    # Run benchmark
    await runner.run_comparative_benchmark(configurations, problems, variant_type)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark AI providers on mathematical problems")
    
    # Benchmark modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Configuration file path")
    group.add_argument("--quick-test", action="store_true", 
                      help="Quick test with 3 problems across all providers")
    
    # Custom benchmark options
    parser.add_argument("--providers", nargs="+", choices=get_supported_providers(),
                       help="Providers to test (for custom benchmark)")
    parser.add_argument("--models", nargs="+", 
                       help="Models to test (for custom benchmark)")
    parser.add_argument("--dataset", type=Path, default="dataset",
                       help="Dataset path (default: dataset)")
    parser.add_argument("--max-problems", type=int, default=10,
                       help="Maximum problems to test (default: 10)")
    parser.add_argument("--variant", default="original",
                       choices=["original", "descriptive_long", "kernel_variant"],
                       help="Problem variant (default: original)")
    parser.add_argument("--output-dir", type=Path, default="benchmark_results",
                       help="Output directory (default: benchmark_results)")
    
    args = parser.parse_args()
    
    try:
        if args.quick_test:
            await run_quick_test()
        elif args.config:
            await run_custom_benchmark(args.config)
        else:
            # Custom benchmark mode (placeholder for future implementation)
            print("Custom benchmark mode not yet implemented. Use --config or --quick-test.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main())) 