#!/usr/bin/env python3
"""
Example of using OpenRouter with putnam-bench to solve mathematical problems.

This example demonstrates:
1. Using different model combinations from different providers
2. Solving a real problem from the dataset
3. Comparing results across different models
"""

import asyncio
import json
import os
from loader import create_loader

async def solve_with_openrouter():
    """Example of solving a Putnam problem using OpenRouter."""
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Load a sample problem
    problem_file = "dataset/1938-A-1.json"
    if not os.path.exists(problem_file):
        print(f"❌ Problem file not found: {problem_file}")
        print("   Make sure you're running from the project root directory")
        return
    
    with open(problem_file) as f:
        problem_data = json.load(f)
    
    print(f"📚 Problem: {problem_data['problem_statement'][:100]}...")
    print(f"   Type: {problem_data['problem_type']}")
    print(f"   Year: {problem_data['year']}")
    
    # Test with different model combinations
    test_configs = [
        {
            "name": "OpenAI Only",
            "solver": "openai/gpt-4o-mini",
            "grader": "openai/gpt-4o"
        },
        {
            "name": "Mixed OpenAI/Anthropic",
            "solver": "openai/gpt-4o",
            "grader": "anthropic/claude-3-haiku"
        },
        {
            "name": "Google Gemini",
            "solver": "google/gemini-pro",
            "grader": "google/gemini-pro"
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {config['name']}")
        print(f"   Solver: {config['solver']}")
        print(f"   Grader: {config['grader']}")
        
        try:
            # Create loader with specific models
            loader = create_loader(
                "openrouter",
                solver_model=config['solver'],
                grader_model=config['grader'],
                retries=3,
                timeout_base=120
            )
            
            # Solve the problem
            print("\n⏳ Solving problem...")
            solution, raw = await loader.solve_problem(problem_data['problem_statement'])
            
            if solution:
                print("✅ Solution found!")
                print(f"   Final answer: {solution.get('final_answer', 'N/A')}")
                
                # Grade the solution (if it's a proof problem)
                if problem_data['problem_type'] == 'proof':
                    print("\n⏳ Grading solution...")
                    grade_result = await loader.grade_solution(
                        problem_data['problem_statement'],
                        solution['solution'],
                        problem_data.get('ground_truth_solution', ''),
                        problem_type='proof'
                    )
                    
                    if grade_result:
                        print(f"📊 Grade: {grade_result.get('score', 'N/A')}/10")
                        print(f"   Reasoning: {grade_result.get('reasoning', 'N/A')[:100]}...")
                else:
                    print("   (Calculation problem - grading skipped)")
            else:
                print("❌ Failed to get solution")
                
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            
    print(f"\n{'='*60}")
    print("✅ Example completed!")

async def list_recommended_models():
    """List recommended model combinations for different use cases."""
    
    print("\n📋 Recommended OpenRouter Model Combinations:\n")
    
    recommendations = [
        {
            "use_case": "Best Quality (Expensive)",
            "solver": "openai/gpt-4o",
            "grader": "anthropic/claude-3-opus",
            "notes": "Highest accuracy but most expensive"
        },
        {
            "use_case": "Balanced Performance",
            "solver": "openai/gpt-4o-mini",
            "grader": "anthropic/claude-3-sonnet",
            "notes": "Good balance of cost and performance"
        },
        {
            "use_case": "Budget Friendly",
            "solver": "openai/gpt-3.5-turbo",
            "grader": "google/gemini-pro",
            "notes": "Cheapest option, still decent quality"
        },
        {
            "use_case": "Open Source Models",
            "solver": "meta-llama/llama-3-70b-instruct",
            "grader": "mistralai/mixtral-8x7b-instruct",
            "notes": "Using open-source models only"
        },
        {
            "use_case": "Code-Focused",
            "solver": "deepseek/deepseek-coder",
            "grader": "meta-llama/codellama-70b-instruct",
            "notes": "Optimized for problems with code"
        }
    ]
    
    for rec in recommendations:
        print(f"🎯 {rec['use_case']}")
        print(f"   Solver: {rec['solver']}")
        print(f"   Grader: {rec['grader']}")
        print(f"   Notes: {rec['notes']}")
        print()

if __name__ == "__main__":
    print("🚀 OpenRouter Example for Putnam Bench")
    
    # Run the example
    asyncio.run(solve_with_openrouter())
    
    # Show recommendations
    asyncio.run(list_recommended_models()) 