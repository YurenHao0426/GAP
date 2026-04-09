#!/usr/bin/env python3
"""
KV on MATH: Generate Kernel Variants for 200 MATH Level 5 problems.
Async parallel with repair loop and 3 judges. Resumes from previous run.
"""

import json, asyncio, random, re, os, sys, time
from datasets import load_dataset
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEM_O3 = asyncio.Semaphore(3)     # o3 calls - conservative
SEM_GPT4O = asyncio.Semaphore(20) # gpt-4o calls
SEM_EVAL = asyncio.Semaphore(40)  # evaluation calls
random.seed(42)

OUTPUT_DIR = '/home/yurenh2/gap/mini_gap_math_results/kv_200'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'kv_generation.json')
LOCK = asyncio.Lock()

# ============================================================
# Prompts
# ============================================================

SLOT_DISCOVERY = """You are a mathematical analysis expert. Given a math problem and its solution, identify all "mutable slots" — numerical constants, parameters, coefficients, or specific values that could be changed to create a new but structurally equivalent problem.

For each slot provide: the original value, what it represents, and constraints on alternatives.

Return ONLY valid JSON:
{"mutable_slots": [{"value": "...", "role": "...", "constraints": "..."}, ...]}
If no mutable slots exist, return: {"mutable_slots": []}"""

BACK_SYNTHESIS = """You are creating a mathematical variant problem. Given an original problem, its solution, and mutable slots:
- Choose NEW values for each slot satisfying constraints
- Rewrite the problem with new values
- Work out the complete new solution step by step
- The new problem MUST be solvable following the same reasoning

Return ONLY valid JSON:
{"new_problem": "...", "new_solution": "...", "new_answer": "...", "slot_changes": [{"original": "...", "new": "..."}]}"""

VERIFY = """You are a rigorous mathematical verifier. Given a problem and solution:
1. Is the problem well-defined and solvable?
2. Is every step mathematically correct?
3. Does the solution correctly arrive at the stated answer?

Reply with EXACTLY:
VERDICT: ACCEPT
or
VERDICT: REJECT
REASON: [what is wrong]"""

REPAIR = """The following mathematical variant was rejected. Fix it.
Problem: {problem}
Solution: {solution}
Rejection reason: {reason}
Return ONLY valid JSON:
{{"new_problem": "...", "new_solution": "...", "new_answer": "..."}}"""

# ============================================================
# API Helpers
# ============================================================

def extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

async def call_api(messages, model="gpt-4o", max_tokens=4000):
    sem = SEM_O3 if model == "o3" else SEM_GPT4O
    async with sem:
        for attempt in range(5):
            try:
                kwargs = {"model": model, "messages": messages}
                if model == "o3":
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["temperature"] = 0
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                wait = min(60, (2 ** attempt) * 3)
                if attempt < 4:
                    await asyncio.sleep(wait)
                else:
                    return None

async def save_result(result, all_results):
    async with LOCK:
        all_results.append(result)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(all_results, f)

# ============================================================
# KV Pipeline
# ============================================================

async def generate_kv(problem, solution, idx, all_results, max_repairs=3, n_judges=3):
    # Stage 1: Slot Discovery (gpt-4o for speed)
    slot_text = await call_api(
        [{"role": "system", "content": SLOT_DISCOVERY},
         {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{solution}"}],
        model="gpt-4o", max_tokens=2000
    )
    slots_data = extract_json(slot_text) if slot_text else None
    if not slots_data or not slots_data.get('mutable_slots'):
        result = {'status': 'no_slots', 'original_index': idx, 'reason': 'no mutable slots'}
        await save_result(result, all_results)
        print(f"[{idx}] no_slots")
        return

    n_slots = len(slots_data['mutable_slots'])

    # Stage 2: Back-synthesis (o3 for quality)
    synth_text = await call_api(
        [{"role": "system", "content": BACK_SYNTHESIS},
         {"role": "user", "content": f"Original problem:\n{problem}\n\nOriginal solution:\n{solution}\n\nMutable slots:\n{json.dumps(slots_data['mutable_slots'])}\n\nCreate a variant."}],
        model="o3", max_tokens=6000
    )
    synth_data = extract_json(synth_text) if synth_text else None
    if not synth_data or not synth_data.get('new_problem'):
        result = {'status': 'error', 'original_index': idx, 'reason': 'synthesis failed'}
        await save_result(result, all_results)
        print(f"[{idx}] synthesis_error")
        return

    new_problem = synth_data['new_problem']
    new_solution = synth_data['new_solution']
    new_answer = synth_data.get('new_answer', '')

    # Stage 3: Verify with repair loop
    for repair_round in range(max_repairs + 1):
        # Run judges in parallel
        judge_tasks = []
        for _ in range(n_judges):
            judge_tasks.append(call_api(
                [{"role": "system", "content": VERIFY},
                 {"role": "user", "content": f"Problem:\n{new_problem}\n\nSolution:\n{new_solution}"}],
                model="o3", max_tokens=500
            ))
        judge_results = await asyncio.gather(*judge_tasks)

        accepts = 0
        reasons = []
        for jr in judge_results:
            if jr and 'ACCEPT' in jr.upper() and 'REJECT' not in jr.upper():
                accepts += 1
            else:
                match = re.search(r'REASON:\s*(.*)', jr or '', re.IGNORECASE)
                reasons.append(match.group(1).strip() if match else (jr or 'unknown')[:200])

        if accepts == n_judges:
            result = {
                'status': 'accepted',
                'original_index': idx,
                'original_problem': problem,
                'original_solution': solution,
                'mutable_slots': slots_data['mutable_slots'],
                'kv_problem': new_problem,
                'kv_solution': new_solution,
                'kv_answer': new_answer,
                'slot_changes': synth_data.get('slot_changes', []),
                'repair_rounds': repair_round,
                'n_slots': n_slots,
            }
            await save_result(result, all_results)
            print(f"[{idx}] ACCEPTED (round {repair_round}, {n_slots} slots)")
            return

        if repair_round < max_repairs:
            reason_str = '; '.join(reasons[:2])[:500]
            repair_text = await call_api(
                [{"role": "system", "content": REPAIR.format(problem=new_problem, solution=new_solution, reason=reason_str)},
                 {"role": "user", "content": "Fix the variant."}],
                model="o3", max_tokens=6000
            )
            repair_data = extract_json(repair_text) if repair_text else None
            if repair_data:
                new_problem = repair_data.get('new_problem', new_problem)
                new_solution = repair_data.get('new_solution', new_solution)
                new_answer = repair_data.get('new_answer', new_answer)

    result = {'status': 'rejected', 'original_index': idx, 'reason': f'failed {max_repairs} repairs'}
    await save_result(result, all_results)
    print(f"[{idx}] REJECTED")

# ============================================================
# Evaluation
# ============================================================

def extract_boxed(text):
    if not text:
        return None
    matches = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        depth = 1; j = idx + 7
        while j < len(text) and depth > 0:
            if text[j] == '{': depth += 1
            elif text[j] == '}': depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[idx+7:j-1].strip())
        i = j
    return matches[-1] if matches else None

async def evaluate_all(accepted_results):
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

    async def grade(ref_answer, student_answer):
        async with SEM_EVAL:
            resp = await client.chat.completions.create(
                model="gpt-4o", temperature=0, max_tokens=10,
                messages=[{"role": "user", "content": f"Are these mathematical answers equivalent? Reference: {ref_answer}\nStudent: {student_answer}\nReply CORRECT or INCORRECT."}],
            )
            text = resp.choices[0].message.content.upper()
            return 'INCORRECT' not in text and 'CORRECT' in text

    eval_models = ['gpt-4o', 'gpt-4o-mini']
    results = {}

    for model in eval_models:
        print(f"\nEvaluating {len(accepted_results)} variants with {model}...")

        # Solve originals
        orig_tasks = [solve(r['original_problem'], model) for r in accepted_results]
        orig_sols = await asyncio.gather(*orig_tasks)

        # Solve KVs
        kv_tasks = [solve(r['kv_problem'], model) for r in accepted_results]
        kv_sols = await asyncio.gather(*kv_tasks)

        # Grade
        orig_grades = []
        kv_grades = []
        for i, r in enumerate(accepted_results):
            ref_orig = extract_boxed(r['original_solution'])
            stu_orig = extract_boxed(orig_sols[i])
            ref_kv = r.get('kv_answer') or extract_boxed(r.get('kv_solution', ''))
            stu_kv = extract_boxed(kv_sols[i])

            og = await grade(ref_orig or 'N/A', stu_orig or 'N/A') if ref_orig and stu_orig else False
            kg = await grade(ref_kv or 'N/A', stu_kv or 'N/A') if ref_kv and stu_kv else False
            orig_grades.append(og)
            kv_grades.append(kg)

        orig_acc = sum(orig_grades) / len(orig_grades) * 100
        kv_acc = sum(kv_grades) / len(kv_grades) * 100

        results[model] = {
            'original_accuracy': orig_acc,
            'kv_accuracy': kv_acc,
            'delta': kv_acc - orig_acc,
            'n': len(accepted_results),
            'orig_correct': sum(orig_grades),
            'kv_correct': sum(kv_grades),
        }
        print(f"  {model}: orig={orig_acc:.1f}%, kv={kv_acc:.1f}%, Δ={kv_acc-orig_acc:+.1f}pp (n={len(accepted_results)})")

    return results

# ============================================================
# Main
# ============================================================

async def main():
    # Load all Level 5 problems
    subsets = ['algebra', 'number_theory', 'precalculus', 'intermediate_algebra', 'counting_and_probability', 'geometry']
    all_level5 = []
    for subset in subsets:
        ds = load_dataset('EleutherAI/hendrycks_math', subset, split='test')
        for item in ds:
            if item.get('level') == 'Level 5' and len(item.get('solution', '')) > 50:
                item['subject'] = subset
                all_level5.append(item)

    random.shuffle(all_level5)
    selected = all_level5[:200]
    print(f"Selected {len(selected)} Level 5 problems")

    # Load previous results
    prev_file = '/home/yurenh2/gap/mini_gap_math_results/kv_50/kv_generation.json'
    prev_accepted = []
    if os.path.exists(prev_file):
        with open(prev_file) as f:
            prev_data = json.load(f)
        prev_accepted = [r for r in prev_data if r['status'] == 'accepted']
        print(f"Loaded {len(prev_accepted)} previously accepted variants")

    # Load current progress
    all_results = []
    done_indices = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            all_results = json.load(f)
        done_indices = {r['original_index'] for r in all_results}
        print(f"Resuming: {len(all_results)} already processed")

    # Generate remaining
    remaining = [(i, p) for i, p in enumerate(selected) if i not in done_indices]
    print(f"Remaining: {len(remaining)} problems")

    # Process in batches of 10 for controlled parallelism
    BATCH_SIZE = 10
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        tasks = [generate_kv(p['problem'], p['solution'], i, all_results) for i, p in batch]
        await asyncio.gather(*tasks)

        # Status update
        from collections import Counter
        status = Counter(r['status'] for r in all_results)
        accepted_count = status.get('accepted', 0)
        print(f"\n--- Progress: {len(all_results)}/200, accepted={accepted_count}, status={dict(status)} ---\n")

    # Combine with previous accepted
    new_accepted = [r for r in all_results if r['status'] == 'accepted']
    all_accepted = prev_accepted + new_accepted
    print(f"\nTotal accepted: {len(all_accepted)} ({len(prev_accepted)} prev + {len(new_accepted)} new)")

    # Evaluate
    if all_accepted:
        print(f"\nEvaluating {len(all_accepted)} accepted KV variants...")
        eval_results = await evaluate_all(all_accepted)

        final = {
            'generation_summary': {
                'total_attempted': 200 + 50,
                'new_accepted': len(new_accepted),
                'prev_accepted': len(prev_accepted),
                'total_accepted': len(all_accepted),
            },
            'evaluation': eval_results,
        }
        with open(os.path.join(OUTPUT_DIR, 'kv_final_results.json'), 'w') as f:
            json.dump(final, f, indent=2)

        print(f"\n{'='*60}")
        print(f"FINAL KV RESULTS ({len(all_accepted)} variants)")
        print(f"{'='*60}")
        for model, res in eval_results.items():
            print(f"  {model}: orig={res['original_accuracy']:.1f}%, kv={res['kv_accuracy']:.1f}%, Δ={res['delta']:+.1f}pp")

asyncio.run(main())
