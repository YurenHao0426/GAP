#!/usr/bin/env python3
"""
Mini-GAP-MATH: Apply GAP surface renaming to MATH dataset and evaluate.
Proves GAP framework generalizes beyond Putnam.
"""

import json
import re
import random
import os
import sys
import time
import argparse
from pathlib import Path

random.seed(42)

# ============================================================
# Step 1: Extract variables from MATH problems
# ============================================================

def extract_latex_variables(problem_text):
    """Extract single-letter and short math variables from LaTeX."""
    # Find variables inside $...$ math mode
    math_segments = re.findall(r'\$([^$]+)\$', problem_text)
    all_text = ' '.join(math_segments)

    # Common math variables: single letters, subscripted versions
    vars_found = set()

    # Single-letter variables (a-z, A-Z) used standalone in math
    for m in re.finditer(r'(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z{])', all_text):
        v = m.group(1)
        # Exclude common function names and constants
        if v not in {'e', 'i', 'd', 'f', 'g', 'h', 'sin', 'cos', 'tan', 'log', 'ln', 'lim', 'max', 'min'}:
            vars_found.add(v)

    # Subscripted variables like x_1, a_n
    for m in re.finditer(r'([a-zA-Z])_\{?([a-zA-Z0-9]+)\}?', all_text):
        vars_found.add(f"{m.group(1)}_{m.group(2)}")

    return list(vars_found)

# ============================================================
# Step 2: Surface renaming - Garbled String (GS) variant
# ============================================================

def generate_garbled_name(length=None):
    """Generate a random alphanumeric string (4-12 chars)."""
    if length is None:
        length = random.randint(4, 12)
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(chars, k=length))

def generate_descriptive_long_name(var_name):
    """Generate a descriptive long confusing name (DLC)."""
    # Pool of unrelated words
    words = [
        'marshmallow', 'butterfly', 'telescope', 'pineapple', 'volcano',
        'watermelon', 'dinosaur', 'moonlight', 'umbrella', 'strawberry',
        'caterpillar', 'sunflower', 'kangaroo', 'chocolate', 'thunderbolt',
        'penguin', 'trampoline', 'avalanche', 'cinnamon', 'dragonfly',
        'elephant', 'fireworks', 'giraffe', 'honeybee', 'igloo',
        'jellyfish', 'kaleidoscope', 'lighthouse', 'mandarin', 'nutmeg',
        'origami', 'platypus', 'quicksilver', 'rainbow', 'saxophone',
        'tumbleweed', 'unicorn', 'velvet', 'whirlpool', 'xylophone'
    ]
    n_words = random.randint(2, 4)
    return ''.join(random.sample(words, n_words))

def apply_surface_rename(problem_text, solution_text, var_map):
    """Apply variable renaming to both problem and solution."""
    new_problem = problem_text
    new_solution = solution_text

    # Sort by length (longest first) to avoid partial replacements
    sorted_vars = sorted(var_map.keys(), key=len, reverse=True)

    for old_var, new_var in [(v, var_map[v]) for v in sorted_vars]:
        # Handle subscripted variables
        if '_' in old_var:
            base, sub = old_var.split('_', 1)
            # Replace in LaTeX: x_{1} or x_1
            patterns = [
                (rf'(?<![a-zA-Z]){re.escape(base)}_\{{{re.escape(sub)}\}}', f'{new_var}'),
                (rf'(?<![a-zA-Z]){re.escape(base)}_{re.escape(sub)}(?![a-zA-Z0-9])', f'{new_var}'),
            ]
            for pat, repl in patterns:
                new_problem = re.sub(pat, repl, new_problem)
                new_solution = re.sub(pat, repl, new_solution)
        else:
            # Single letter variable - be careful with context
            # Replace inside math mode ($...$) only
            def replace_in_math(text, old, new):
                def replacer(match):
                    content = match.group(1)
                    # Replace standalone variable
                    content = re.sub(
                        rf'(?<![a-zA-Z\\]){re.escape(old)}(?![a-zA-Z])',
                        new, content
                    )
                    return f'${content}$'
                return re.sub(r'\$([^$]+)\$', replacer, text)

            new_problem = replace_in_math(new_problem, old_var, new_var)
            new_solution = replace_in_math(new_solution, old_var, new_var)

    return new_problem, new_solution

def create_variants(problems):
    """Create GS and DLC variants for each problem."""
    results = []
    for idx, prob in enumerate(problems):
        variables = extract_latex_variables(prob['problem'])
        if len(variables) == 0:
            # No variables to rename, skip
            continue

        # Create variable mappings
        used_gs = set()
        used_dlc = set()
        gs_map = {}
        dlc_map = {}

        for v in variables:
            # Garbled String
            gs_name = generate_garbled_name()
            while gs_name in used_gs:
                gs_name = generate_garbled_name()
            used_gs.add(gs_name)
            gs_map[v] = gs_name

            # Descriptive Long Confusing
            dlc_name = generate_descriptive_long_name(v)
            while dlc_name in used_dlc:
                dlc_name = generate_descriptive_long_name(v)
            used_dlc.add(dlc_name)
            dlc_map[v] = dlc_name

        gs_problem, gs_solution = apply_surface_rename(
            prob['problem'], prob['solution'], gs_map
        )
        dlc_problem, dlc_solution = apply_surface_rename(
            prob['problem'], prob['solution'], dlc_map
        )

        results.append({
            'index': idx,
            'subject': prob.get('subject', 'unknown'),
            'level': prob.get('level', 'unknown'),
            'original': {
                'problem': prob['problem'],
                'solution': prob['solution'],
            },
            'garbled_string': {
                'problem': gs_problem,
                'solution': gs_solution,
                'map': gs_map,
            },
            'descriptive_long_confusing': {
                'problem': dlc_problem,
                'solution': dlc_solution,
                'map': dlc_map,
            },
            'variables': variables,
        })

    return results


# ============================================================
# Step 3: Evaluation with local models
# ============================================================

def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} in MATH-style solutions."""
    # Find the last \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    # Remove \$ signs
    ans = ans.replace('$', '')
    # Remove spaces
    ans = ans.replace(' ', '')
    # Normalize fractions
    ans = ans.replace('\\dfrac', '\\frac')
    ans = ans.replace('\\tfrac', '\\frac')
    return ans

def check_answer(generated, reference_solution):
    """Check if generated answer matches reference."""
    ref_answer = extract_boxed_answer(reference_solution)
    gen_answer = extract_boxed_answer(generated)

    if ref_answer is None or gen_answer is None:
        return False

    return normalize_answer(ref_answer) == normalize_answer(gen_answer)


def run_inference_batch(model, tokenizer, problems, device, batch_size=4):
    """Run inference on a batch of problems."""
    import torch

    results = []
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = []
        for p in batch:
            messages = [
                {"role": "system", "content": "You are an expert mathematician. Solve the problem step by step and put your final answer in \\boxed{}."},
                {"role": "user", "content": p}
            ]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted = f"Problem: {p}\n\nSolve step by step. Put final answer in \\boxed{{}}.\n\nSolution:"
            prompts.append(formatted)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.extend([g.strip() for g in generated])

        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {min(i+batch_size, len(problems))}/{len(problems)}")

    return results


def evaluate_model(model_name, variants_data, device="cuda:2", batch_size=4):
    """Evaluate a single model on original + variants."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    dtype = torch.float16 if 'cuda' in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype
    )
    model.eval()

    results = {'model': model_name, 'variants': {}}

    for variant_type in ['original', 'garbled_string', 'descriptive_long_confusing']:
        print(f"\n--- Evaluating {variant_type} ---")

        problems = [item[variant_type]['problem'] for item in variants_data]
        solutions = [item[variant_type]['solution'] for item in variants_data]

        generated = run_inference_batch(model, tokenizer, problems, device, batch_size)

        correct = 0
        total = len(problems)
        per_item = []
        for j, (gen, sol) in enumerate(zip(generated, solutions)):
            is_correct = check_answer(gen, sol)
            correct += int(is_correct)
            per_item.append({
                'index': variants_data[j]['index'],
                'correct': is_correct,
                'generated_answer': extract_boxed_answer(gen),
                'reference_answer': extract_boxed_answer(sol),
            })

        acc = correct / total * 100 if total > 0 else 0
        results['variants'][variant_type] = {
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'per_item': per_item,
        }
        print(f"  {variant_type}: {correct}/{total} = {acc:.1f}%")

    # Compute deltas
    orig_acc = results['variants']['original']['accuracy']
    for vt in ['garbled_string', 'descriptive_long_confusing']:
        var_acc = results['variants'][vt]['accuracy']
        results['variants'][vt]['delta'] = var_acc - orig_acc

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Mini-GAP-MATH experiment')
    parser.add_argument('--step', choices=['prepare', 'evaluate', 'all'], default='all')
    parser.add_argument('--models', nargs='+', default=['Qwen/Qwen2.5-7B-Instruct'])
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-problems', type=int, default=200)
    parser.add_argument('--input', default='/home/yurenh2/gap/math_sample_200.json')
    parser.add_argument('--output-dir', default='/home/yurenh2/gap/mini_gap_math_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    variants_file = os.path.join(args.output_dir, 'math_variants.json')

    if args.step in ['prepare', 'all']:
        print("="*60)
        print("Step 1: Loading MATH problems and creating variants")
        print("="*60)

        with open(args.input) as f:
            problems = json.load(f)

        problems = problems[:args.max_problems]
        print(f"Loaded {len(problems)} problems")

        variants = create_variants(problems)
        print(f"Created variants for {len(variants)} problems")

        with open(variants_file, 'w') as f:
            json.dump(variants, f, indent=2)
        print(f"Saved to {variants_file}")

        # Show a sample
        if variants:
            v = variants[0]
            print(f"\nSample problem (original):")
            print(f"  {v['original']['problem'][:200]}...")
            print(f"  Variables: {v['variables']}")
            print(f"\nGS variant:")
            print(f"  {v['garbled_string']['problem'][:200]}...")
            print(f"  Map: {v['garbled_string']['map']}")

    if args.step in ['evaluate', 'all']:
        print("\n" + "="*60)
        print("Step 2: Evaluating models")
        print("="*60)

        with open(variants_file) as f:
            variants_data = json.load(f)

        all_results = []
        for model_name in args.models:
            try:
                result = evaluate_model(
                    model_name, variants_data,
                    device=args.device, batch_size=args.batch_size
                )
                all_results.append(result)

                # Save incrementally
                out_file = os.path.join(args.output_dir, 'evaluation_results.json')
                with open(out_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"ERROR with {model_name}: {e}")
                import traceback
                traceback.print_exc()

        # Print summary table
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"{'Model':<35} {'Original':>10} {'GS':>10} {'GS Δ':>8} {'DLC':>10} {'DLC Δ':>8}")
        print("-"*85)
        for r in all_results:
            m = r['model'].split('/')[-1]
            orig = r['variants']['original']['accuracy']
            gs = r['variants']['garbled_string']['accuracy']
            gs_d = r['variants']['garbled_string']['delta']
            dlc = r['variants']['descriptive_long_confusing']['accuracy']
            dlc_d = r['variants']['descriptive_long_confusing']['delta']
            print(f"{m:<35} {orig:>9.1f}% {gs:>9.1f}% {gs_d:>+7.1f} {dlc:>9.1f}% {dlc_d:>+7.1f}")


if __name__ == '__main__':
    main()
