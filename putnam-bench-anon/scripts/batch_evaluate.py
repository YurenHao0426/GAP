#!/usr/bin/env python3
"""
Batch evaluation script for processing entire datasets with multiple providers.

This script efficiently processes all JSON files in the dataset directory,
supports multiple AI providers, and generates comprehensive evaluation reports.

Features:
- Incremental saving: Results are saved after each problem completes
- Simple resume support: Skip already completed problems based on checkpoint
- Multi-provider support
- Comprehensive evaluation reports

Usage:
    python batch_evaluate.py --provider openai --output results/openai_results.json
    python batch_evaluate.py --provider anthropic --variant kernel_variant --max-concurrent 5
    
Resume usage (simplified):
    # Resume with same configuration
    python batch_evaluate.py --provider openai --dataset dataset/ --resume checkpoint_file.json
    
    # Resume with different settings (checkpoint only provides skip list)
    python batch_evaluate.py --provider openai --dataset dataset/ --concurrent 10 --resume checkpoint_file.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path
import argparse
from typing import List, Dict, Any
import logging
from datetime import datetime
import shutil

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar
    class tqdm:
        def __init__(self, total=None, desc=None, **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc
            print(f"{desc}: Starting...")
        
        def update(self, n=1):
            self.n += n
            if self.total:
                percent = (self.n / self.total) * 100
                print(f"{self.desc}: {self.n}/{self.total} ({percent:.1f}%)", end='\r')
        
        def set_postfix(self, postfix_dict):
            pass
        
        def close(self):
            print()  # New line after progress

# Add the loader module to the path
sys.path.append(str(Path(__file__).parent))

from loader import create_loader, get_supported_providers


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


async def load_dataset(dataset_path: Path, max_files: int = None) -> List[Dict[str, Any]]:
    """Load all JSON files from the dataset directory."""
    json_files = list(dataset_path.glob("*.json"))
    
    if max_files:
        json_files = json_files[:max_files]
    
    problems = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_source_file'] = str(json_file.name)
                problems.append(data)
        except Exception as e:
            logging.warning(f"Failed to load {json_file}: {str(e)}")
    
    return problems


async def process_single_problem(loader, problem_data: Dict[str, Any], 
                                variant_type: str, solver_model: str = None, 
                                grader_model: str = None) -> Dict[str, Any]:
    """Process a single problem and return results with metadata."""
    start_time = time.time()
    
    try:
        result = await loader.test_single_problem(
            problem_data,
            variant_type=variant_type,
            solver_model=solver_model,
            grader_model=grader_model
        )
        
        # Add metadata
        result['_metadata'] = {
            'source_file': problem_data.get('_source_file', 'unknown'),
            'variant_type': variant_type,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'models_used': {
                'solver': solver_model or loader.solver_model,
                'grader': grader_model or loader.grader_model
            }
        }
        
        return result
        
    except Exception as e:
        # Return error information
        return {
            'error': str(e),
            'final_grade': 0,
            '_metadata': {
                'source_file': problem_data.get('_source_file', 'unknown'),
                'variant_type': variant_type,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
        }


async def batch_evaluate(dataset_path: Path = None, provider: str = None, variant_type: str = "original",
                        max_concurrent: int = 3, max_files: int = None,
                        solver_model: str = None, grader_model: str = None,
                        output_file: Path = None, resume_checkpoint: Path = None,
                        **loader_kwargs) -> Dict[str, Any]:
    """
    Batch evaluate problems using specified provider with resume support.
    
    Args:
        dataset_path: Path to dataset directory (required for new runs or old checkpoint format)
        provider: AI provider name (required for new runs or old checkpoint format)
        variant_type: Problem variant to use
        max_concurrent: Maximum concurrent evaluations
        max_files: Maximum number of files to process (None for all)
        solver_model: Override solver model
        grader_model: Override grader model
        output_file: Output file path
        resume_checkpoint: Path to checkpoint file to resume from
        **loader_kwargs: Additional arguments for loader
        
    Returns:
        Dictionary with evaluation results and statistics
    """
    logger = logging.getLogger(__name__)
    
    # Check if resuming from checkpoint
    if resume_checkpoint and resume_checkpoint.exists():
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Simple resume: just restore completed indices and results
        completed_indices = set(checkpoint_data.get('completed_indices', []))
        results = checkpoint_data.get('results', [])
        failed_indices = checkpoint_data.get('failed_indices', [])
        successful_indices = checkpoint_data.get('successful_indices', [])
        correct_indices = checkpoint_data.get('correct_indices', [])
        
        # Always require dataset_path and provider from command line
        if not dataset_path:
            raise ValueError("dataset_path is required when resuming")
        if not provider:
            raise ValueError("provider is required when resuming")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        problems = await load_dataset(dataset_path, max_files)
        logger.info(f"Loaded {len(problems)} problems")
        
        if not problems:
            raise ValueError("No problems found in dataset")
        
        checkpoint_file = resume_checkpoint  # Continue using the same checkpoint file
        logger.info(f"Resuming with {len(completed_indices)} completed problems out of {len(problems)}")
    else:
        # New evaluation - validate required parameters
        if not dataset_path:
            raise ValueError("dataset_path is required for new evaluation")
        if not provider:
            raise ValueError("provider is required for new evaluation")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        problems = await load_dataset(dataset_path, max_files)
        logger.info(f"Loaded {len(problems)} problems")
        
        if not problems:
            raise ValueError("No problems found in dataset")
        
        # Initialize state for new run
        completed_indices = set()
        results = []
        failed_indices = []
        successful_indices = []
        correct_indices = []
        
        # Create checkpoint file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_file:
            checkpoint_file = output_file.parent / f"checkpoint_{output_file.stem}_{timestamp}.json"
        else:
            checkpoint_file = Path(f"checkpoint_{provider}_{variant_type}_{timestamp}.json")
    
    # Create loader
    logger.info(f"Creating {provider} loader")
    
    # Include solver_model and grader_model in loader_kwargs if specified
    if solver_model:
        loader_kwargs['solver_model'] = solver_model
    if grader_model:
        loader_kwargs['grader_model'] = grader_model
    
    loader = create_loader(provider, **loader_kwargs)
    
    # Health check
    logger.info("Performing health check...")
    if not await loader.health_check():
        raise RuntimeError(f"Health check failed for {provider}")
    
    # Cost estimation
    logger.info("Estimating costs...")
    cost_info = await loader.estimate_cost(len(problems))
    logger.info(f"Estimated cost: ${cost_info.get('total_cost', 0):.2f}")
    
    # Progress tracking
    remaining_problems = [p for p in problems if p.get('index', 'unknown') not in completed_indices]
    progress_bar = tqdm(total=len(problems), desc=f"Evaluating with {provider}", initial=len(completed_indices))
    
    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    def save_checkpoint():
        """Save current state to checkpoint file - simplified version"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            # Only save essential state information
            'completed_indices': list(completed_indices),
            'successful_indices': successful_indices,
            'failed_indices': failed_indices,
            'correct_indices': correct_indices,
            'results': results,
            # Save minimal config for reference (not for resume)
            'dataset_path': str(dataset_path),  # For convenience
            'total_problems': len(problems),
            'current_config': {
                'provider': provider,
                'variant_type': variant_type,
                'solver_model': loader.solver_model,
                'grader_model': loader.grader_model
            }
        }
        
        # Write to temporary file first, then move (atomic operation)
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.replace(checkpoint_file)
    
    async def evaluate_problem(problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single problem with concurrency control."""
        problem_index = problem_data.get('index', 'unknown')
        
        # Skip if already completed
        if problem_index in completed_indices:
            return None
        
        async with semaphore:
            try:
                result = await loader.test_single_problem(
                    problem_data,
                    variant_type=variant_type
                )
                
                # Track success/failure based on technical completion, not correctness
                if result.get('status') == 'completed':
                    successful_indices.append(result['index'])  # Successfully processed
                    if result.get('correct'):
                        correct_indices.append(result['index'])  # Also correct
                else:
                    failed_indices.append(result['index'])  # Technical failure
                
                # Add to results and mark as completed
                results.append(result)
                completed_indices.add(problem_index)
                
                # Save checkpoint immediately after each problem
                save_checkpoint()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'success': len(successful_indices),
                    'failed': len(failed_indices),
                    'saved': len(completed_indices)
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Error evaluating problem {problem_index}: {e}")
                result = {
                    'index': problem_index,
                    'status': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                
                # Add to results and mark as completed (even if failed)
                results.append(result)
                failed_indices.append(problem_index)
                completed_indices.add(problem_index)
                
                # Save checkpoint
                save_checkpoint()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'success': len(successful_indices),
                    'failed': len(failed_indices),
                    'saved': len(completed_indices)
                })
                
                return result
    
    # Run evaluations
    start_time = time.time()
    
    try:
        # Create tasks only for remaining problems
        tasks = [evaluate_problem(problem) for problem in remaining_problems]
        
        if tasks:
            # Execute all tasks concurrently (limited by semaphore)
            await asyncio.gather(*tasks)
        else:
            logger.info("All problems already completed!")
    
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Progress saved to checkpoint.")
        logger.info(f"To resume, use: --resume {checkpoint_file}")
        raise
    
    finally:
        progress_bar.close()
    
    # Calculate statistics
    total_time = time.time() - start_time
    completed_results = [r for r in results if r.get('status') == 'completed']
    grades = [r['grade']['grade'] for r in completed_results 
              if r.get('grade', {}).get('status') == 'success' and 'grade' in r.get('grade', {})]
    
    # Calculate numeric grades (CORRECT=5, INCORRECT=2.5)
    numeric_grades = [5.0 if g == 'CORRECT' else 2.5 for g in grades]
    average_grade = sum(numeric_grades) / len(numeric_grades) if numeric_grades else 0.0
    
    summary = {
        'total_problems': len(problems),
        'completed': len(completed_results),
        'successful': len(successful_indices),  # Technical success (completed processing)
        'failed': len(failed_indices),  # Technical failures
        'correct_answers': len(correct_indices),  # Mathematically correct answers
        'incorrect_answers': len(successful_indices) - len(correct_indices),  # Wrong but processed
        'success_rate': (len(successful_indices) / len(problems) * 100) if problems else 0,  # Technical success rate
        'accuracy_rate': (len(correct_indices) / len(successful_indices) * 100) if successful_indices else 0,  # Correctness rate
        'average_grade': average_grade,
        'total_time_seconds': total_time,
        'problems_per_second': len(problems) / total_time if total_time > 0 else 0,
        'provider': provider,
        'variant_type': variant_type,
        'solver_model': loader.solver_model,
        'grader_model': loader.grader_model,
        'max_concurrent': max_concurrent,
        'estimated_cost': cost_info,
        'checkpoint_file': str(checkpoint_file)
    }
    
    # Create full results
    full_results = {
        'summary': summary,
        'problems': results,
        'successful_indices': successful_indices,  # Technical successes
        'failed_indices': failed_indices,  # Technical failures
        'correct_indices': correct_indices,  # Correct answers
        'timestamp': datetime.now().isoformat()
    }
    
    # Save final results
    if output_file:
        logger.info(f"Saving final results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # Clean up checkpoint file after successful completion
        if checkpoint_file.exists():
            logger.info(f"Removing checkpoint file: {checkpoint_file}")
            checkpoint_file.unlink()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Provider: {provider}")
    logger.info(f"Variant: {variant_type}")
    logger.info(f"Total problems: {summary['total_problems']}")
    logger.info(f"✅ Successfully processed: {summary['successful']} ({summary['success_rate']:.1f}%)")
    logger.info(f"💥 Technical failures: {summary['failed']}")
    logger.info(f"🎯 Correct answers: {summary['correct_answers']} ({summary['accuracy_rate']:.1f}% of processed)")
    logger.info(f"❌ Wrong answers: {summary['incorrect_answers']}")
    logger.info(f"Average grade: {summary['average_grade']:.2f}")
    logger.info(f"Total time: {summary['total_time_seconds']:.1f}s")
    logger.info(f"Speed: {summary['problems_per_second']:.2f} problems/second")
    
    # Cleanup
    if hasattr(loader, '__aexit__'):
        await loader.__aexit__(None, None, None)
    
    return full_results


async def batch_evaluate_cross(dataset_path: Path = None, 
                             solver_provider: str = None,
                             grader_provider: str = None,
                             variant_type: str = "original",
                             max_concurrent: int = 3, 
                             max_files: int = None,
                             solver_model: str = None, 
                             grader_model: str = None,
                             output_file: Path = None,
                             resume_checkpoint: Path = None,
                             vllm_url: str = None,
                             device: str = None,
                             quick: bool = False) -> Dict[str, Any]:
    """
    Batch evaluate problems using different providers for solving and grading with resume support.
    
    Args:
        dataset_path: Path to dataset directory (required for new runs, ignored for resume)
        solver_provider: Provider for solving problems (required for new runs, ignored for resume)
        grader_provider: Provider for grading (if None, uses solver_provider)
        variant_type: Problem variant to use
        max_concurrent: Maximum concurrent evaluations
        max_files: Maximum number of files to process (None for all)
        solver_model: Override solver model
        grader_model: Override grader model
        output_file: Output file path
        resume_checkpoint: Path to checkpoint file to resume from
        vllm_url: VLLM server URL if using VLLM
        device: Device for HuggingFace models
        
    Returns:
        Dictionary with evaluation results and statistics
    """
    logger = logging.getLogger(__name__)
    
    # Check if resuming from checkpoint
    if resume_checkpoint and resume_checkpoint.exists():
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Simple resume: just restore completed indices and results
        completed_indices = set(checkpoint_data.get('completed_indices', []))
        results = checkpoint_data.get('results', [])
        failed_indices = checkpoint_data.get('failed_indices', [])
        successful_indices = checkpoint_data.get('successful_indices', [])
        correct_indices = checkpoint_data.get('correct_indices', [])
        
        # Always require providers and dataset_path from command line
        if not dataset_path:
            raise ValueError("dataset_path is required when resuming")
        if not solver_provider:
            raise ValueError("solver_provider is required when resuming")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        problems = await load_dataset(dataset_path, max_files)
        logger.info(f"Loaded {len(problems)} problems")
        
        if not problems:
            raise ValueError("No problems found in dataset")
        
        checkpoint_file = resume_checkpoint  # Continue using the same checkpoint file
        logger.info(f"Resuming with {len(completed_indices)} completed problems out of {len(problems)}")
    else:
        # New evaluation - validate required parameters
        if not dataset_path:
            raise ValueError("dataset_path is required for new evaluation")
        if not solver_provider:
            raise ValueError("solver_provider is required for new evaluation")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        problems = await load_dataset(dataset_path, max_files)
        logger.info(f"Loaded {len(problems)} problems")
        
        if not problems:
            raise ValueError("No problems found in dataset")
        
        # Initialize state for new run
        completed_indices = set()
        results = []
        failed_indices = []
        successful_indices = []
        correct_indices = []
        
        # Create checkpoint file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_file:
            checkpoint_file = output_file.parent / f"checkpoint_{output_file.stem}_{timestamp}.json"
        else:
            checkpoint_file = Path(f"checkpoint_cross_{solver_provider}_{grader_provider or solver_provider}_{variant_type}_{timestamp}.json")
    
    # Create cross-provider loader
    logger.info(f"Creating cross-provider loader: solver={solver_provider}, grader={grader_provider or solver_provider}")
    
    from loader import create_cross_provider_loader
    
    # Prepare kwargs for each provider
    loader_kwargs = {}
    
    # VLLM settings
    if vllm_url:
        if solver_provider == 'vllm':
            loader_kwargs['solver_kwargs'] = {'base_url': vllm_url}
        if grader_provider == 'vllm':
            loader_kwargs['grader_kwargs'] = {'base_url': vllm_url}
    
    # HuggingFace settings
    if device:
        if solver_provider == 'huggingface':
            loader_kwargs['solver_kwargs'] = {'device': device}
        if grader_provider == 'huggingface':
            loader_kwargs['grader_kwargs'] = {'device': device}
    
    # Add quick mode if specified
    if quick:
        loader_kwargs['quick'] = True
    
    loader = create_cross_provider_loader(
        solver_provider=solver_provider,
        grader_provider=grader_provider,
        solver_model=solver_model,
        grader_model=grader_model,
        **loader_kwargs
    )
    
    # Health check
    logger.info("Performing health check...")
    if not await loader.health_check():
        raise RuntimeError(f"Health check failed")
    
    # Cost estimation
    logger.info("Estimating costs...")
    cost_info = await loader.estimate_cost(len(problems))
    logger.info(f"Estimated cost: ${cost_info.get('total_cost', 0):.2f}")
    
    # Progress tracking
    remaining_problems = [p for p in problems if p.get('index', 'unknown') not in completed_indices]
    progress_bar = tqdm(total=len(problems), desc=f"Evaluating (solver={solver_provider}, grader={grader_provider or solver_provider})", initial=len(completed_indices))
    
    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    def save_checkpoint():
        """Save current state to checkpoint file - simplified version"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            # Only save essential state information
            'completed_indices': list(completed_indices),
            'successful_indices': successful_indices,
            'failed_indices': failed_indices,
            'correct_indices': correct_indices,
            'results': results,
            # Save minimal config for reference (not for resume)
            'dataset_path': str(dataset_path),  # For convenience
            'total_problems': len(problems),
            'current_config': {
                'solver_provider': solver_provider,
                'grader_provider': grader_provider or solver_provider,
                'variant_type': variant_type,
                'solver_model': loader.solver_model,
                'grader_model': loader.grader_model
            }
        }
        
        # Write to temporary file first, then move (atomic operation)
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.replace(checkpoint_file)
    
    async def evaluate_problem(problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single problem with concurrency control."""
        problem_index = problem_data.get('index', 'unknown')
        
        # Skip if already completed
        if problem_index in completed_indices:
            return None
        
        async with semaphore:
            try:
                result = await loader.test_single_problem(
                    problem_data,
                    variant_type=variant_type
                )
                
                # Track success/failure based on technical completion, not correctness
                if result.get('status') == 'completed':
                    successful_indices.append(result['index'])  # Successfully processed
                    if result.get('correct'):
                        correct_indices.append(result['index'])  # Also correct
                else:
                    failed_indices.append(result['index'])  # Technical failure
                
                # Add to results and mark as completed
                results.append(result)
                completed_indices.add(problem_index)
                
                # Save checkpoint immediately after each problem
                save_checkpoint()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'success': len(successful_indices),
                    'failed': len(failed_indices),
                    'saved': len(completed_indices)
                })
                
                return result
                
            except Exception as e:
                import traceback
                
                # Capture full error details
                error_details = {
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat(),
                    'problem_index': problem_index,
                    'problem_title': problem_data.get('title', 'unknown')
                }
                
                # Try to capture HTTP-specific details if available
                if hasattr(e, 'response'):
                    try:
                        error_details['http_status'] = e.response.status_code
                        error_details['http_headers'] = dict(e.response.headers)
                        error_details['http_response_text'] = e.response.text
                    except:
                        pass
                
                # Try to capture request details if available
                if hasattr(e, 'request'):
                    try:
                        error_details['request_method'] = e.request.method
                        error_details['request_url'] = e.request.url
                        error_details['request_headers'] = dict(e.request.headers)
                        # Don't log request body as it might contain sensitive info
                    except:
                        pass
                
                # Log detailed error
                logger.error(f"DETAILED ERROR for problem {problem_index}:")
                logger.error(f"  Error Type: {error_details['error_type']}")
                logger.error(f"  Error Message: {error_details['error_message']}")
                logger.error(f"  Problem Title: {error_details['problem_title']}")
                
                if 'http_status' in error_details:
                    logger.error(f"  HTTP Status: {error_details['http_status']}")
                    logger.error(f"  HTTP Response: {error_details['http_response_text'][:500]}...")
                
                logger.error(f"  Full Traceback:\n{error_details['traceback']}")
                
                # Save to detailed error log
                error_log_file = output_file.parent / f"detailed_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" if output_file else Path(f"detailed_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                
                try:
                    # Load existing errors if file exists
                    if error_log_file.exists():
                        with open(error_log_file, 'r') as f:
                            existing_errors = json.load(f)
                    else:
                        existing_errors = []
                    
                    # Add new error
                    existing_errors.append(error_details)
                    
                    # Save updated errors
                    with open(error_log_file, 'w') as f:
                        json.dump(existing_errors, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Detailed error saved to {error_log_file}")
                    
                except Exception as save_error:
                    logger.error(f"Failed to save detailed error log: {save_error}")
                
                result = {
                    'index': problem_index,
                    'status': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'error_details': error_details
                }
                
                # Add to results and mark as completed (even if failed)
                results.append(result)
                failed_indices.append(problem_index)
                completed_indices.add(problem_index)
                
                # Save checkpoint
                save_checkpoint()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'success': len(successful_indices),
                    'failed': len(failed_indices),
                    'saved': len(completed_indices)
                })
                
                return result
    
    # Run evaluations
    start_time = time.time()
    
    try:
        # Create tasks only for remaining problems
        tasks = [evaluate_problem(problem) for problem in remaining_problems]
        
        if tasks:
            # Execute all tasks concurrently (limited by semaphore)
            await asyncio.gather(*tasks)
        else:
            logger.info("All problems already completed!")
    
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Progress saved to checkpoint.")
        logger.info(f"To resume, use: --resume {checkpoint_file}")
        raise
    
    finally:
        progress_bar.close()
    
    # Calculate statistics
    total_time = time.time() - start_time
    completed_results = [r for r in results if r.get('status') == 'completed']
    grades = [r['grade']['grade'] for r in completed_results 
              if r.get('grade', {}).get('status') == 'success' and 'grade' in r.get('grade', {})]
    
    # Calculate numeric grades (CORRECT=5, INCORRECT=2.5)
    numeric_grades = [5.0 if g == 'CORRECT' else 2.5 for g in grades]
    average_grade = sum(numeric_grades) / len(numeric_grades) if numeric_grades else 0.0
    
    model_info = loader.get_model_info()
    
    summary = {
        'total_problems': len(problems),
        'completed': len(completed_results),
        'successful': len(successful_indices),  # Technical success (completed processing)
        'failed': len(failed_indices),  # Technical failures
        'correct_answers': len(correct_indices),  # Mathematically correct answers
        'incorrect_answers': len(successful_indices) - len(correct_indices),  # Wrong but processed
        'success_rate': (len(successful_indices) / len(problems) * 100) if problems else 0,  # Technical success rate
        'accuracy_rate': (len(correct_indices) / len(successful_indices) * 100) if successful_indices else 0,  # Correctness rate
        'average_grade': average_grade,
        'total_time_seconds': total_time,
        'problems_per_second': len(problems) / total_time if total_time > 0 else 0,
        'solver_provider': model_info.get('solver_provider', solver_provider),
        'grader_provider': model_info.get('grader_provider', grader_provider or solver_provider),
        'variant_type': variant_type,
        'solver_model': loader.solver_model,
        'grader_model': loader.grader_model,
        'max_concurrent': max_concurrent,
        'estimated_cost': cost_info,
        'is_cross_provider': model_info.get('is_cross_provider', False)
    }
    
    # Create full results
    full_results = {
        'summary': summary,
        'problems': results,
        'successful_indices': successful_indices,  # Technical successes
        'failed_indices': failed_indices,  # Technical failures
        'correct_indices': correct_indices,  # Correct answers
        'timestamp': datetime.now().isoformat()
    }
    
    # Save if requested
    if output_file:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-PROVIDER EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Solver Provider: {summary['solver_provider']} ({loader.solver_model})")
    logger.info(f"Grader Provider: {summary['grader_provider']} ({loader.grader_model})")
    logger.info(f"Variant: {variant_type}")
    logger.info(f"Total problems: {summary['total_problems']}")
    logger.info(f"✅ Successfully processed: {summary['successful']} ({summary['success_rate']:.1f}%)")
    logger.info(f"💥 Technical failures: {summary['failed']}")
    logger.info(f"🎯 Correct answers: {summary['correct_answers']} ({summary['accuracy_rate']:.1f}% of processed)")
    logger.info(f"❌ Wrong answers: {summary['incorrect_answers']}")
    logger.info(f"Average grade: {summary['average_grade']:.2f}")
    logger.info(f"Total time: {summary['total_time_seconds']:.1f}s")
    logger.info(f"Speed: {summary['problems_per_second']:.2f} problems/second")
    
    # Cleanup
    if hasattr(loader, '__aexit__'):
        await loader.__aexit__(None, None, None)
    
    return full_results


async def batch_evaluate_all_variants(dataset_path: Path, provider: str, 
                                     variants: List[str] = None,
                                     max_concurrent: int = 3, max_files: int = None,
                                     solver_model: str = None, grader_model: str = None,
                                     output_dir: Path = None, 
                                     base_url: str = None, device: str = None) -> Dict[str, Any]:
    """
    Batch evaluate problems across all variants using specified provider.
    
    Args:
        dataset_path: Path to dataset directory
        provider: AI provider name
        variants: List of variants to test (None for all)
        max_concurrent: Maximum concurrent evaluations
        max_files: Maximum number of files to process per variant (None for all)
        solver_model: Override solver model
        grader_model: Override grader model
        output_dir: Output directory path
        **loader_kwargs: Additional arguments for loader
        
    Returns:
        Dictionary with all variant results and comparative analysis
    """
    if variants is None:
        variants = ["original", "descriptive_long", "descriptive_long_confusing", 
                   "descriptive_long_misleading", "garbled_string", "kernel_variant"]
    
    if output_dir is None:
        output_dir = Path("results")
    
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_name = f"{provider}"
    if solver_model:
        config_name += f"_{solver_model.replace('/', '_').replace('-', '_')}"
    
    # Create configuration-specific output directory
    config_output_dir = output_dir / f"{config_name}_{timestamp}"
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare loader kwargs based on provider
    loader_kwargs = {}
    if provider == 'vllm' and base_url:
        loader_kwargs['base_url'] = base_url
    elif provider == 'huggingface' and device:
        loader_kwargs['device'] = device
    
    logger.info(f"🚀 Starting multi-variant test for {config_name}")
    logger.info(f"📊 Testing {len(variants)} variants with up to {max_files or 'ALL'} files each")
    
    overall_start_time = time.time()
    variant_results = {}
    
    # Create overall progress bar for variants if tqdm is available
    if HAS_TQDM:
        variant_progress = tqdm.tqdm(total=len(variants), desc="Variants", 
                                   unit="variant", position=1, leave=True)
    
    for i, variant in enumerate(variants):
        logger.info(f"\n📝 [{i+1}/{len(variants)}] Testing variant: {variant}")
        variant_start_time = time.time()
        
        # Output file for this variant
        variant_output_file = config_output_dir / f"{variant}_{timestamp}.json"
        
        try:
            # Run batch evaluation for this variant
            result = await batch_evaluate(
                dataset_path=dataset_path,
                provider=provider,
                variant_type=variant,
                max_concurrent=max_concurrent,
                max_files=max_files,
                solver_model=solver_model,
                grader_model=grader_model,
                output_file=variant_output_file,
                **loader_kwargs
            )
            
            variant_time = time.time() - variant_start_time
            
            # Extract key metrics
            summary = result.get('summary', {})
            variant_results[variant] = {
                'status': 'success',
                'output_file': str(variant_output_file),
                'total_problems': summary.get('total_problems', 0),
                'successful_evaluations': summary.get('successful', 0),
                'correct_evaluations': summary.get('correct_answers', 0),
                'incorrect_evaluations': summary.get('incorrect_answers', 0),
                'failed_evaluations': summary.get('failed', 0),
                'success_rate': summary.get('success_rate', 0),
                'average_grade': summary.get('average_grade', 0),
                'total_processing_time': summary.get('total_time_seconds', 0),
                'avg_time_per_problem': summary.get('problems_per_second', 0),
                'variant_test_time': variant_time,
                'grade_distribution': result.get('problems', []) # Assuming 'problems' contains all results
            }
            
            logger.info(f"✅ {variant}: "
                       f"Grade {summary.get('average_grade', 0):.2f}, "
                       f"Success {summary.get('success_rate', 0):.1f}%, "
                       f"Time {variant_time/60:.1f}min")
            
        except Exception as e:
            variant_time = time.time() - variant_start_time
            error_msg = str(e)
            
            variant_results[variant] = {
                'status': 'failed',
                'error': error_msg,
                'variant_test_time': variant_time
            }
            
            logger.error(f"❌ {variant} failed: {error_msg}")
        
        # Update variant progress bar
        if HAS_TQDM and 'variant_progress' in locals():
            variant_progress.update(1)
            successful_variants_count = len([v for v, r in variant_results.items() if r.get('status') == 'success'])
            variant_progress.set_postfix({
                'Success': successful_variants_count,
                'Failed': len(variant_results) - successful_variants_count
            })
    
    # Close variant progress bar
    if HAS_TQDM and 'variant_progress' in locals():
        variant_progress.close()
    
    overall_time = time.time() - overall_start_time
    
    # Generate comprehensive summary
    successful_variants = [v for v, r in variant_results.items() if r.get('status') == 'success']
    failed_variants = [v for v, r in variant_results.items() if r.get('status') == 'failed']
    
    # Calculate aggregate statistics
    if successful_variants:
        total_problems = sum(variant_results[v].get('total_problems', 0) for v in successful_variants)
        total_successful = sum(variant_results[v].get('successful_evaluations', 0) for v in successful_variants)
        total_correct = sum(variant_results[v].get('correct_evaluations', 0) for v in successful_variants)
        total_incorrect = sum(variant_results[v].get('incorrect_evaluations', 0) for v in successful_variants)
        total_failed = sum(variant_results[v].get('failed_evaluations', 0) for v in successful_variants)
        
        grades = [variant_results[v].get('average_grade', 0) for v in successful_variants]
        success_rates = [variant_results[v].get('success_rate', 0) for v in successful_variants]
        times = [variant_results[v].get('avg_time_per_problem', 0) for v in successful_variants]
        
        overall_avg_grade = sum(grades) / len(grades) if grades else 0
        overall_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        overall_avg_time = sum(times) / len(times) if times else 0
        
        # Find best and worst performing variants
        best_variant = max(successful_variants, key=lambda v: variant_results[v].get('average_grade', 0))
        worst_variant = min(successful_variants, key=lambda v: variant_results[v].get('average_grade', 0))
        
        fastest_variant = min(successful_variants, key=lambda v: variant_results[v].get('avg_time_per_problem', float('inf')))
        slowest_variant = max(successful_variants, key=lambda v: variant_results[v].get('avg_time_per_problem', 0))
    else:
        total_problems = total_successful = total_correct = total_incorrect = total_failed = 0
        overall_avg_grade = overall_success_rate = overall_avg_time = 0
        best_variant = worst_variant = fastest_variant = slowest_variant = None
    
    summary_result = {
        'configuration': {
            'provider': provider,
            'solver_model': solver_model,
            'grader_model': grader_model,
            'base_url': base_url,
            'device': device,
            'timestamp': timestamp
        },
        'test_overview': {
            'total_variants_tested': len(variant_results),
            'successful_variants': len(successful_variants),
            'failed_variants': len(failed_variants),
            'total_test_time_minutes': overall_time / 60,
            'variants_list': list(variant_results.keys())
        },
        'aggregate_metrics': {
            'total_problems_across_variants': total_problems,
            'total_successful_evaluations': total_successful,
            'total_correct_evaluations': total_correct,
            'total_incorrect_evaluations': total_incorrect,
            'total_technical_failures': total_failed,
            'overall_average_grade': overall_avg_grade,
            'overall_success_rate': overall_success_rate,
            'overall_avg_time_per_problem': overall_avg_time
        },
        'variant_comparison': {
            'best_performing_variant': {
                'variant': best_variant,
                'grade': variant_results.get(best_variant, {}).get('average_grade', 0) if best_variant else 0
            },
            'worst_performing_variant': {
                'variant': worst_variant,
                'grade': variant_results.get(worst_variant, {}).get('average_grade', 0) if worst_variant else 0
            },
            'fastest_variant': {
                'variant': fastest_variant,
                'time_per_problem': variant_results.get(fastest_variant, {}).get('avg_time_per_problem', 0) if fastest_variant else 0
            },
            'slowest_variant': {
                'variant': slowest_variant,
                'time_per_problem': variant_results.get(slowest_variant, {}).get('avg_time_per_problem', 0) if slowest_variant else 0
            }
        },
        'detailed_variant_results': variant_results
    }
    
    # Save configuration summary
    summary_file = config_output_dir / f"SUMMARY_{config_name}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)
    
    # Print summary to console
    logger.info("\n" + "="*80)
    logger.info("📊 MULTI-VARIANT TEST SUMMARY REPORT")
    logger.info("="*80)
    
    logger.info(f"🤖 Provider: {provider}")
    if solver_model:
        logger.info(f"🧠 Solver Model: {solver_model}")
    if grader_model:
        logger.info(f"📝 Grader Model: {grader_model}")
    
    logger.info(f"\n📋 Test Overview:")
    logger.info(f"   Total variants tested: {len(variant_results)}")
    logger.info(f"   Successful variants: {len(successful_variants)}")
    logger.info(f"   Failed variants: {len(failed_variants)}")
    logger.info(f"   Total test time: {overall_time/60:.1f} minutes")
    
    if total_problems > 0:
        logger.info(f"\n📈 Aggregate Performance:")
        logger.info(f"   Total problems: {total_problems}")
        logger.info(f"   Overall average grade: {overall_avg_grade:.2f}")
        logger.info(f"   Overall success rate: {overall_success_rate:.1f}%")
        logger.info(f"   Average time per problem: {overall_avg_time:.2f}s")
    
    if best_variant:
        logger.info(f"\n🏆 Variant Performance:")
        logger.info(f"   Best performing: {best_variant} (Grade: {variant_results[best_variant]['average_grade']:.2f})")
        logger.info(f"   Worst performing: {worst_variant} (Grade: {variant_results[worst_variant]['average_grade']:.2f})")
        logger.info(f"   Fastest: {fastest_variant} ({variant_results[fastest_variant]['avg_time_per_problem']:.2f}s/problem)")
        logger.info(f"   Slowest: {slowest_variant} ({variant_results[slowest_variant]['avg_time_per_problem']:.2f}s/problem)")
    
    logger.info("="*80)
    logger.info(f"💾 Configuration summary saved to {summary_file}")
    
    return summary_result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch evaluate mathematical problems")
    
    # Required arguments
    parser.add_argument("--provider", required=True, choices=get_supported_providers(),
                       help="AI provider to use")
    
    # Dataset options
    parser.add_argument("--dataset", default="dataset", 
                       help="Dataset directory path (default: dataset)")
    parser.add_argument("--variant", default="original",
                       choices=["original", "descriptive_long", "descriptive_long_confusing", 
                               "descriptive_long_misleading", "garbled_string", "kernel_variant"],
                       help="Problem variant to use (default: original)")
    parser.add_argument("--all-variants", action="store_true",
                       help="Test all 6 problem variants instead of just one")
    parser.add_argument("--variants", nargs="+",
                       choices=["original", "descriptive_long", "descriptive_long_confusing", 
                               "descriptive_long_misleading", "garbled_string", "kernel_variant"],
                       help="Specific variants to test (use with --all-variants)")
    parser.add_argument("--max-files", type=int,
                       help="Maximum number of files to process per variant (default: all)")
    
    # Processing options
    parser.add_argument("--max-concurrent", type=int, default=3,
                       help="Maximum concurrent evaluations (default: 3)")
    parser.add_argument("--solver-model", 
                       help="Override solver model")
    parser.add_argument("--grader-model",
                       help="Override grader model")
    
    # Output options
    parser.add_argument("--output", type=Path,
                       help="Output file path (default: results/[provider]_[timestamp].json)")
    parser.add_argument("--output-dir", type=Path, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--resume", type=Path,
                       help="Path to checkpoint file to resume from")
    
    # Provider-specific options
    parser.add_argument("--base-url", 
                       help="Base URL for VLLM provider")
    parser.add_argument("--device", default="auto",
                       help="Device for HuggingFace provider (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    # Default output file if not specified
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = args.output_dir / f"{args.provider}_{args.variant}_{timestamp}.json"
    
    # Prepare loader kwargs based on provider
    loader_kwargs = {}
    if args.provider == 'vllm' and args.base_url:
        loader_kwargs['base_url'] = args.base_url
    elif args.provider == 'huggingface' and args.device:
        loader_kwargs['device'] = args.device
    
    try:
        if args.all_variants or args.variants:
            # Multi-variant evaluation
            variants_to_test = args.variants if args.variants else None
            results = await batch_evaluate_all_variants(
                dataset_path=Path(args.dataset),
                provider=args.provider,
                variants=variants_to_test,
                max_concurrent=args.max_concurrent,
                max_files=args.max_files,
                solver_model=args.solver_model,
                grader_model=args.grader_model,
                output_dir=args.output_dir,
                base_url=args.base_url,
                device=args.device
            )
            
            logger.info(f"Multi-variant evaluation completed successfully!")
            logger.info(f"Overall average grade: {results['aggregate_metrics']['overall_average_grade']:.2f}")
            logger.info(f"Overall success rate: {results['aggregate_metrics']['overall_success_rate']:.1f}%")
        else:
            # Single variant evaluation
            results = await batch_evaluate(
                dataset_path=Path(args.dataset),
                provider=args.provider,
                variant_type=args.variant,
                max_concurrent=args.max_concurrent,
                max_files=args.max_files,
                solver_model=args.solver_model,
                grader_model=args.grader_model,
                output_file=args.output,
                resume_checkpoint=args.resume,
                **loader_kwargs
            )
            
            logger.info(f"Batch evaluation completed successfully!")
            logger.info(f"Average grade: {results['summary']['average_grade']:.2f}")
            logger.info(f"Success rate: {results['summary']['success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main())) 