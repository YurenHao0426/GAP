#!/usr/bin/env python3
"""
Mini-GAP-MATH: Evaluate MATH variants using OpenAI API.
"""

import json
import re
import os
import sys
import asyncio
import time
import argparse
from pathlib import Path
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEMAPHORE = asyncio.Semaphore(50)  # max concurrent requests

# ============================================================
# Answer extraction and checking
# ============================================================

def extract_boxed_answer(text):
    """Extract answer from \\boxed{...}."""
    if not text:
        return None
    # Handle nested braces
    matches = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 1
        j = idx + 7
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[idx+7:j-1].strip())
        i = j
    return matches[-1] if matches else None

def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace('$', '').replace(' ', '')
    ans = ans.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = ans.replace('\\,', '').replace('\\;', '')
    return ans

def check_answer(generated, reference_solution):
    """Check if generated answer matches reference."""
    ref_answer = extract_boxed_answer(reference_solution)
    gen_answer = extract_boxed_answer(generated)
    if ref_answer is None or gen_answer is None:
        return False
    return normalize_answer(ref_answer) == normalize_answer(gen_answer)

# ============================================================
# API calls
# ============================================================

async def solve_problem(problem_text, model="gpt-4o-mini"):
    """Solve a single problem using OpenAI API."""
    async with SEMAPHORE:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert mathematician. Solve the problem step by step and put your final answer in \\boxed{}."},
                    {"role": "user", "content": problem_text}
                ],
                max_tokens=2048,
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API error: {e}")
            return None

async def evaluate_variant(variant_data, variant_type, model):
    """Evaluate all problems for one variant type."""
    problems = [item[variant_type]['problem'] for item in variant_data]
    solutions = [item[variant_type]['solution'] for item in variant_data]

    print(f"\n--- Evaluating {variant_type} ({len(problems)} problems) ---")

    # Launch all requests concurrently
    tasks = [solve_problem(p, model) for p in problems]
    generated = await asyncio.gather(*tasks)

    correct = 0
    total = len(problems)
    per_item = []
    for j, (gen, sol) in enumerate(zip(generated, solutions)):
        is_correct = check_answer(gen or "", sol)
        correct += int(is_correct)
        per_item.append({
            'index': variant_data[j]['index'],
            'correct': is_correct,
            'generated_answer': extract_boxed_answer(gen or ""),
            'reference_answer': extract_boxed_answer(sol),
        })

    acc = correct / total * 100 if total > 0 else 0
    print(f"  {variant_type}: {correct}/{total} = {acc:.1f}%")

    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'per_item': per_item,
    }

async def evaluate_model(model, variant_data, output_dir):
    """Evaluate a model on all variants."""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model}")
    print(f"{'='*60}")

    results = {'model': model, 'variants': {}}

    for vt in ['original', 'garbled_string', 'descriptive_long_confusing']:
        results['variants'][vt] = await evaluate_variant(variant_data, vt, model)

    # Compute deltas
    orig_acc = results['variants']['original']['accuracy']
    for vt in ['garbled_string', 'descriptive_long_confusing']:
        results['variants'][vt]['delta'] = results['variants'][vt]['accuracy'] - orig_acc

    # Save
    out_file = os.path.join(output_dir, f'{model.replace("/", "_")}_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_file}")

    return results

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt-4o-mini'])
    parser.add_argument('--variants-file', default='/home/yurenh2/gap/mini_gap_math_results/math_variants.json')
    parser.add_argument('--output-dir', default='/home/yurenh2/gap/mini_gap_math_results')
    parser.add_argument('--concurrency', type=int, default=50)
    args = parser.parse_args()

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(args.concurrency)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.variants_file) as f:
        variant_data = json.load(f)

    print(f"Loaded {len(variant_data)} problems with variants")

    all_results = []
    for model in args.models:
        result = await evaluate_model(model, variant_data, args.output_dir)
        all_results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("MINI-GAP-MATH RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Original':>10} {'GS':>10} {'GS Δ':>8} {'DLC':>10} {'DLC Δ':>8}")
    print("-"*75)
    for r in all_results:
        m = r['model']
        orig = r['variants']['original']['accuracy']
        gs = r['variants']['garbled_string']['accuracy']
        gs_d = r['variants']['garbled_string']['delta']
        dlc = r['variants']['descriptive_long_confusing']['accuracy']
        dlc_d = r['variants']['descriptive_long_confusing']['delta']
        print(f"{m:<25} {orig:>9.1f}% {gs:>9.1f}% {gs_d:>+7.1f} {dlc:>9.1f}% {dlc_d:>+7.1f}")

    # Save combined
    combined_file = os.path.join(args.output_dir, 'all_api_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_file}")

if __name__ == '__main__':
    asyncio.run(main())
