#!/usr/bin/env python3
"""
KV on MATH: Generate Kernel Variants for 50 MATH Level 5 problems.
3-stage pipeline with repair loop and 3 judges.
"""

import json, asyncio, random, re, os, sys, time
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEM_GEN = asyncio.Semaphore(2)   # generation calls (expensive o3) - low to avoid rate limits
SEM_EVAL = asyncio.Semaphore(30)  # evaluation calls (cheaper)
random.seed(42)

OUTPUT_DIR = '/home/yurenh2/gap/mini_gap_math_results/kv_50'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Prompts
# ============================================================

SLOT_DISCOVERY = """You are a mathematical analysis expert. Given a math problem and its solution, identify all "mutable slots" — numerical constants, parameters, coefficients, or specific values that could be changed to create a new but structurally equivalent problem.

For each slot:
1. The original value (as it appears in the problem)
2. What it represents mathematically
3. Constraints on alternative values (what must hold for the same solution technique to work)

Return ONLY valid JSON:
{
  "mutable_slots": [
    {"value": "...", "role": "...", "constraints": "..."},
    ...
  ]
}

If the problem has no mutable slots (all constants are mathematically fixed), return:
{"mutable_slots": []}"""

BACK_SYNTHESIS = """You are creating a mathematical variant problem. Given:
1. An original problem and solution
2. Mutable slots identified in the problem

Your task:
- Choose NEW values for each mutable slot that satisfy the constraints
- Rewrite the problem with these new values
- Work out the complete new solution step by step
- The new problem MUST be solvable and the solution MUST follow the same mathematical reasoning

Return ONLY valid JSON:
{
  "new_problem": "... (full LaTeX problem statement)",
  "new_solution": "... (complete step-by-step solution)",
  "new_answer": "... (final answer that would go in \\boxed{})",
  "slot_changes": [{"original": "...", "new": "..."}]
}"""

VERIFY = """You are a rigorous mathematical verifier. Given a problem and its proposed solution:

1. Is the problem well-defined and solvable?
2. Is every step in the solution mathematically correct?
3. Does the solution correctly arrive at the stated answer?

If ANY step is wrong or the answer doesn't follow, you MUST reject.

Reply with EXACTLY:
VERDICT: ACCEPT
or
VERDICT: REJECT
REASON: [what is wrong]"""

REPAIR = """The following mathematical variant was rejected by a verifier.

Problem: {problem}
Solution: {solution}
Rejection reason: {reason}

Please fix the solution (or the problem if needed) so that it is mathematically correct.
Keep the same problem structure and slot changes.

Return ONLY valid JSON:
{{"new_problem": "...", "new_solution": "...", "new_answer": "..."}}"""

# ============================================================
# KV Generation Pipeline
# ============================================================

def extract_json(text):
    """Extract JSON from text that may contain markdown or extra text."""
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Try finding JSON block
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    # Try finding raw JSON
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


async def call_llm(system_prompt, user_content, max_tokens=4000, model="gpt-4o"):
    """Call LLM with rate limiting and exponential backoff."""
    sem = SEM_GEN if model == "o3" else SEM_EVAL
    async with sem:
        for attempt in range(4):
            try:
                kwargs = {"model": model, "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]}
                if model == "o3":
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["temperature"] = 0
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                wait = (2 ** attempt) * 3
                print(f"    {model} error (attempt {attempt+1}): {e}, waiting {wait}s...")
                await asyncio.sleep(wait)
        print(f"    {model} failed after 4 attempts")
        return None

async def call_o3(system_prompt, user_content, max_tokens=4000):
    return await call_llm(system_prompt, user_content, max_tokens, model="o3")


async def verify_once(problem, solution):
    """Single verification judge call."""
    async with SEM_GEN:
        try:
            resp = await client.chat.completions.create(
                model="o3", max_completion_tokens=500,
                messages=[
                    {"role": "system", "content": VERIFY},
                    {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{solution}"}
                ],
            )
            text = resp.choices[0].message.content
            accepted = 'ACCEPT' in text.upper() and 'REJECT' not in text.upper()
            reason = ""
            if not accepted:
                match = re.search(r'REASON:\s*(.*)', text, re.IGNORECASE)
                reason = match.group(1).strip() if match else text
            return accepted, reason
        except Exception as e:
            return False, str(e)


async def generate_kv_with_repair(problem, solution, idx, max_repairs=3, n_judges=3):
    """Generate KV with repair loop and multiple judges."""
    print(f"\n[{idx}] Processing...")

    # Stage 1: Slot Discovery (use gpt-4o for speed)
    slot_text = await call_llm(SLOT_DISCOVERY, f"Problem:\n{problem}\n\nSolution:\n{solution}", 2000, model="gpt-4o")
    if not slot_text:
        return {'status': 'error', 'reason': 'slot discovery failed'}

    slots_data = extract_json(slot_text)
    if not slots_data or not slots_data.get('mutable_slots'):
        print(f"[{idx}] No mutable slots found")
        return {'status': 'no_slots', 'reason': 'no mutable slots identified'}

    n_slots = len(slots_data['mutable_slots'])
    print(f"[{idx}] Found {n_slots} slots")

    # Stage 2: Back-synthesis (use o3 for quality)
    synth_text = await call_llm(
        BACK_SYNTHESIS,
        f"Original problem:\n{problem}\n\nOriginal solution:\n{solution}\n\nMutable slots:\n{json.dumps(slots_data['mutable_slots'], indent=2)}\n\nCreate a variant with different values.",
        6000, model="o3"
    )
    if not synth_text:
        return {'status': 'error', 'reason': 'back-synthesis failed'}

    synth_data = extract_json(synth_text)
    if not synth_data or not synth_data.get('new_problem'):
        return {'status': 'error', 'reason': 'could not parse synthesized variant'}

    new_problem = synth_data['new_problem']
    new_solution = synth_data['new_solution']
    new_answer = synth_data.get('new_answer', '')

    # Stage 3: Verification with repair loop
    for repair_round in range(max_repairs + 1):
        # Run n_judges in parallel
        judge_tasks = [verify_once(new_problem, new_solution) for _ in range(n_judges)]
        verdicts = await asyncio.gather(*judge_tasks)

        accepts = sum(1 for v, _ in verdicts if v)
        reasons = [r for v, r in verdicts if not v]

        if accepts == n_judges:
            print(f"[{idx}] ACCEPTED (round {repair_round}, {n_judges}/{n_judges} judges)")
            return {
                'status': 'accepted',
                'original_problem': problem,
                'original_solution': solution,
                'mutable_slots': slots_data['mutable_slots'],
                'kv_problem': new_problem,
                'kv_solution': new_solution,
                'kv_answer': new_answer,
                'slot_changes': synth_data.get('slot_changes', []),
                'repair_rounds': repair_round,
                'judge_count': n_judges,
            }

        if repair_round < max_repairs:
            print(f"[{idx}] Round {repair_round}: {accepts}/{n_judges} accepted, repairing...")
            # Repair
            reason_str = '; '.join(reasons[:2])
            repair_text = await call_llm(
                REPAIR.format(problem=new_problem, solution=new_solution, reason=reason_str),
                "Fix the variant.", 6000, model="o3"
            )
            if repair_text:
                repair_data = extract_json(repair_text)
                if repair_data:
                    new_problem = repair_data.get('new_problem', new_problem)
                    new_solution = repair_data.get('new_solution', new_solution)
                    new_answer = repair_data.get('new_answer', new_answer)

    print(f"[{idx}] REJECTED after {max_repairs} repairs")
    return {
        'status': 'rejected',
        'original_problem': problem,
        'reason': f'failed after {max_repairs} repair rounds',
        'last_reasons': reasons,
    }


async def evaluate_kv(accepted_results, variants_data):
    """Evaluate accepted KV variants with GPT-4o and GPT-4o-mini."""
    if not accepted_results:
        return {}

    async def solve(problem, model):
        async with SEM_EVAL:
            resp = await client.chat.completions.create(
                model=model, temperature=0, max_tokens=2048,
                messages=[
                    {"role": "system", "content": "Solve step by step. Put final answer in \\boxed{}."},
                    {"role": "user", "content": problem}
                ],
            )
            return resp.choices[0].message.content

    async def grade(ref_answer, student_answer, model="gpt-4o"):
        async with SEM_EVAL:
            resp = await client.chat.completions.create(
                model=model, temperature=0, max_tokens=10,
                messages=[{"role": "user", "content": f"Are these two mathematical answers equivalent? Reference: {ref_answer}\nStudent: {student_answer}\nReply CORRECT or INCORRECT."}],
            )
            return 'CORRECT' in resp.choices[0].message.content.upper()

    def extract_boxed(text):
        if not text: return None
        matches = []
        i = 0
        while i < len(text):
            idx = text.find('\\boxed{', i)
            if idx == -1: break
            depth = 1; j = idx + 7
            while j < len(text) and depth > 0:
                if text[j] == '{': depth += 1
                elif text[j] == '}': depth -= 1
                j += 1
            if depth == 0: matches.append(text[idx+7:j-1].strip())
            i = j
        return matches[-1] if matches else None

    eval_models = ['gpt-4o', 'gpt-4o-mini']
    results = {}

    for model in eval_models:
        print(f"\nEvaluating with {model}...")

        # Solve originals
        orig_tasks = [solve(r['original_problem'], model) for r in accepted_results]
        orig_solutions = await asyncio.gather(*orig_tasks)

        # Solve KV variants
        kv_tasks = [solve(r['kv_problem'], model) for r in accepted_results]
        kv_solutions = await asyncio.gather(*kv_tasks)

        # Grade originals
        orig_grades = []
        for i, (sol, r) in enumerate(zip(orig_solutions, accepted_results)):
            ref = extract_boxed(r['original_solution'])
            stu = extract_boxed(sol)
            if ref and stu:
                g = await grade(ref, stu)
                orig_grades.append(g)
            else:
                orig_grades.append(False)

        # Grade KVs
        kv_grades = []
        for i, (sol, r) in enumerate(zip(kv_solutions, accepted_results)):
            ref = r['kv_answer'] or extract_boxed(r['kv_solution'])
            stu = extract_boxed(sol)
            if ref and stu:
                g = await grade(ref, stu)
                kv_grades.append(g)
            else:
                kv_grades.append(False)

        orig_acc = sum(orig_grades) / len(orig_grades) * 100
        kv_acc = sum(kv_grades) / len(kv_grades) * 100
        delta = kv_acc - orig_acc

        results[model] = {
            'original_accuracy': orig_acc,
            'kv_accuracy': kv_acc,
            'delta': delta,
            'n': len(accepted_results),
            'orig_correct': sum(orig_grades),
            'kv_correct': sum(kv_grades),
        }
        print(f"  {model}: orig={orig_acc:.1f}%, kv={kv_acc:.1f}%, Δ={delta:+.1f}pp")

    return results


async def main():
    # Load MATH Level 5 problems
    with open('/home/yurenh2/gap/math_sample_200.json') as f:
        all_problems = json.load(f)

    level5 = [p for p in all_problems if p['level'] == 'Level 5' and len(p['solution']) > 50]
    selected = random.sample(level5, min(50, len(level5)))
    print(f"Selected {len(selected)} Level 5 problems for KV generation")

    # Load any previous progress
    progress_file = os.path.join(OUTPUT_DIR, 'kv_generation.json')
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            kv_results = json.load(f)
        done_indices = {r['original_index'] for r in kv_results}
        print(f"Resuming: {len(kv_results)} already done")
    else:
        kv_results = []
        done_indices = set()

    # Generate KVs sequentially to avoid rate limits
    for i, p in enumerate(selected):
        if i in done_indices:
            continue
        result = await generate_kv_with_repair(
            p['problem'], p['solution'], i,
            max_repairs=3, n_judges=3
        )
        result['original_index'] = i
        result['subject'] = p.get('subject', 'unknown')
        kv_results.append(result)

        # Save incrementally
        with open(progress_file, 'w') as f:
            json.dump(kv_results, f, indent=2)

        # Small delay between problems
        await asyncio.sleep(2)

    # Summary
    accepted = [r for r in kv_results if r['status'] == 'accepted']
    rejected = [r for r in kv_results if r['status'] == 'rejected']
    no_slots = [r for r in kv_results if r['status'] == 'no_slots']
    errors = [r for r in kv_results if r['status'] == 'error']

    print(f"\n{'='*60}")
    print(f"KV GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempted: {len(selected)}")
    print(f"Accepted: {len(accepted)} ({len(accepted)/len(selected)*100:.0f}%)")
    print(f"Rejected: {len(rejected)} ({len(rejected)/len(selected)*100:.0f}%)")
    print(f"No slots: {len(no_slots)} ({len(no_slots)/len(selected)*100:.0f}%)")
    print(f"Errors: {len(errors)} ({len(errors)/len(selected)*100:.0f}%)")

    if accepted:
        avg_repairs = sum(r['repair_rounds'] for r in accepted) / len(accepted)
        print(f"Avg repair rounds (accepted): {avg_repairs:.1f}")

    # Evaluate accepted variants
    if accepted:
        print(f"\n{'='*60}")
        print(f"EVALUATING {len(accepted)} ACCEPTED KV VARIANTS")
        print(f"{'='*60}")
        eval_results = await evaluate_kv(accepted, None)

        # Save everything
        final_results = {
            'generation_summary': {
                'total': len(selected),
                'accepted': len(accepted),
                'rejected': len(rejected),
                'no_slots': len(no_slots),
                'errors': len(errors),
            },
            'evaluation': eval_results,
            'accepted_variants': accepted,
        }
        with open(os.path.join(OUTPUT_DIR, 'kv_final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Orig%':>8} {'KV%':>8} {'Δ':>8} {'N':>5}")
        print("-"*50)
        for model, res in eval_results.items():
            print(f"{model:<20} {res['original_accuracy']:>7.1f}% {res['kv_accuracy']:>7.1f}% {res['delta']:>+7.1f} {res['n']:>5}")

asyncio.run(main())
