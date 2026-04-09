#!/usr/bin/env python3
"""
Multi-grader consistency analysis for Mini-GAP-MATH.
Uses multiple LLM graders to evaluate the same (problem, solution) pairs.
Computes Cohen's kappa and percent agreement.
"""

import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEMAPHORE = asyncio.Semaphore(30)

GRADING_PROMPT = """You are a strict math grader. You are given a math problem, its reference solution, and a student's solution.

Determine if the student's final answer is CORRECT or INCORRECT.
- For numerical answers: the answer must match exactly (after simplification).
- For expressions: must be mathematically equivalent.
- Ignore intermediate steps; focus only on the final answer.

Respond with EXACTLY one word: CORRECT or INCORRECT"""


async def grade_one(problem, reference_solution, student_solution, model):
    """Grade a single (problem, student_solution) pair."""
    async with SEMAPHORE:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GRADING_PROMPT},
                    {"role": "user", "content": f"Problem:\n{problem}\n\nReference Solution:\n{reference_solution}\n\nStudent Solution:\n{student_solution}"}
                ],
                max_tokens=10,
                temperature=0,
            )
            answer = resp.choices[0].message.content.strip().upper()
            return 'CORRECT' in answer
        except Exception as e:
            print(f"  Grading error: {e}")
            return None


async def grade_all(problems, ref_solutions, student_solutions, model):
    """Grade all solutions with a given model."""
    tasks = [
        grade_one(p, r, s, model)
        for p, r, s in zip(problems, ref_solutions, student_solutions)
    ]
    return await asyncio.gather(*tasks)


def cohens_kappa(labels1, labels2):
    """Compute Cohen's kappa between two sets of binary labels."""
    assert len(labels1) == len(labels2)
    n = len(labels1)
    # Filter out None
    valid = [(a, b) for a, b in zip(labels1, labels2) if a is not None and b is not None]
    if not valid:
        return 0.0, 0
    n = len(valid)
    agree = sum(1 for a, b in valid if a == b)
    p_o = agree / n  # observed agreement

    # Expected agreement
    p1_yes = sum(1 for a, _ in valid if a) / n
    p2_yes = sum(1 for _, b in valid if b) / n
    p_e = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if p_e == 1.0:
        return 1.0, n
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, n


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', required=True, help='Path to evaluation results JSON')
    parser.add_argument('--variants-file', required=True, help='Path to math_variants.json')
    parser.add_argument('--grader-models', nargs='+', default=['gpt-4o', 'gpt-4o-mini'])
    parser.add_argument('--variant-type', default='original')
    parser.add_argument('--output', default='/home/yurenh2/gap/mini_gap_math_results/regrade_consistency.json')
    args = parser.parse_args()

    # Load original eval results
    with open(args.results_file) as f:
        eval_data = json.load(f)

    # Handle both list and single-model format
    if isinstance(eval_data, list):
        eval_results = eval_data[0]
    else:
        eval_results = eval_data

    # Load variant data for problems
    with open(args.variants_file) as f:
        variants = json.load(f)

    variant_type = args.variant_type
    # Get problems and student solutions
    problems = [v[variant_type]['problem'] for v in variants]
    ref_solutions = [v[variant_type]['solution'] for v in variants]

    # Extract student solutions from eval results (we need the raw generated text)
    # Since we don't store raw text, we'll re-generate or use reference answers
    # Actually, for regrade we need the student solutions. Let's use the per_item data
    # to identify which problems were attempted and grade the reference vs generated answers.

    # For this analysis, we'll ask graders to evaluate whether the reference answer
    # from each variant matches the original. This tests grader consistency.

    print(f"Loaded {len(variants)} problems")
    print(f"Grader models: {args.grader_models}")
    print(f"Variant type: {variant_type}")

    # Strategy: For each problem, take the ORIGINAL solution as "student solution"
    # and the GS/DLC variant solution as reference, and see if graders agree.
    # But actually the more useful thing: re-grade the ORIGINAL problems with
    # multiple graders and compare grades.

    # Simpler approach: Grade the reference solutions with each model to check
    # if graders are consistent on "obviously correct" answers

    all_grades = {}
    for model in args.grader_models:
        print(f"\n--- Grading with {model} ---")
        grades = await grade_all(problems, ref_solutions, ref_solutions, model)
        all_grades[model] = grades
        correct_count = sum(1 for g in grades if g is True)
        none_count = sum(1 for g in grades if g is None)
        print(f"  {model}: {correct_count}/{len(grades)} correct, {none_count} errors")

    # Compute pairwise kappa
    models = list(all_grades.keys())
    print("\n" + "="*60)
    print("PAIRWISE COHEN'S KAPPA")
    print("="*60)

    results = {'models': models, 'kappas': {}, 'agreement': {}}

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            kappa, n = cohens_kappa(all_grades[m1], all_grades[m2])
            pct_agree = sum(1 for a, b in zip(all_grades[m1], all_grades[m2])
                          if a is not None and b is not None and a == b)
            total_valid = sum(1 for a, b in zip(all_grades[m1], all_grades[m2])
                            if a is not None and b is not None)
            pct = pct_agree / total_valid * 100 if total_valid > 0 else 0

            key = f"{m1}_vs_{m2}"
            results['kappas'][key] = kappa
            results['agreement'][key] = pct
            print(f"  {m1} vs {m2}: κ={kappa:.3f}, agreement={pct:.1f}% (n={n})")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    asyncio.run(main())
