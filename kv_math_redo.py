#!/usr/bin/env python3
"""
KV redo: Re-run slot discovery with o3 on no_slots problems, then evaluate all accepted.
"""
import json, asyncio, random, re, os
from datasets import load_dataset
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEM_O3 = asyncio.Semaphore(3)
SEM_EVAL = asyncio.Semaphore(40)
random.seed(42)

OUTPUT_DIR = '/home/yurenh2/gap/mini_gap_math_results/kv_200'
REDO_FILE = os.path.join(OUTPUT_DIR, 'kv_redo.json')
LOCK = asyncio.Lock()

# Prompts
SLOT_DISCOVERY_O3 = """You are a world-class mathematician. Given a math problem and its reference solution, find ALL numerical constants, coefficients, parameters, or specific values that could be changed to create a structurally equivalent but numerically different problem.

Be AGGRESSIVE in finding slots. Even if a value seems "natural" (like 2, 3, etc.), if changing it to another value would still yield a solvable problem with the same solution technique, list it.

Examples of mutable slots:
- Coefficients in equations (2x+3 → 5x+7)
- Exponents (x^3 → x^5)
- Bounds or limits (sum from 1 to 100 → sum from 1 to 200)
- Specific numbers in word problems
- Dimensions or sizes
- Modular bases

Return ONLY valid JSON:
{"mutable_slots": [{"value": "...", "role": "...", "constraints": "..."}, ...]}
If truly no slots exist (every constant is mathematically forced), return: {"mutable_slots": []}"""

BACK_SYNTHESIS = """You are creating a mathematical variant. Given the original problem, solution, and mutable slots:
- Choose NEW values satisfying constraints
- Rewrite the full problem with new values
- Solve it completely step by step
- The solution MUST use the same mathematical technique

Return ONLY valid JSON:
{"new_problem": "...", "new_solution": "...", "new_answer": "...", "slot_changes": [{"original": "...", "new": "..."}]}"""

VERIFY = """You are a rigorous mathematical verifier. Check:
1. Is the problem well-defined?
2. Is every solution step correct?
3. Does it reach the stated answer?

Reply EXACTLY: VERDICT: ACCEPT or VERDICT: REJECT
REASON: [explanation]"""

REPAIR = """Fix this rejected variant.
Problem: {problem}
Solution: {solution}
Reason: {reason}
Return ONLY JSON: {{"new_problem": "...", "new_solution": "...", "new_answer": "..."}}"""

def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

async def api_call(messages, model="o3", max_tokens=4000):
    sem = SEM_O3 if model == "o3" else SEM_EVAL
    async with sem:
        for attempt in range(5):
            try:
                kw = {"model": model, "messages": messages}
                if model == "o3": kw["max_completion_tokens"] = max_tokens
                else: kw["max_tokens"] = max_tokens; kw["temperature"] = 0
                r = await client.chat.completions.create(**kw)
                return r.choices[0].message.content
            except Exception as e:
                w = min(60, (2**attempt)*3)
                if attempt < 4: await asyncio.sleep(w)
                else: return None

async def save(result, results_list):
    async with LOCK:
        results_list.append(result)
        with open(REDO_FILE, 'w') as f:
            json.dump(results_list, f)

async def process_one(problem, solution, idx, results_list):
    # Stage 1: o3 slot discovery
    slot_text = await api_call(
        [{"role": "system", "content": SLOT_DISCOVERY_O3},
         {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{solution}"}],
        model="o3", max_tokens=2000)
    slots = extract_json(slot_text) if slot_text else None
    if not slots or not slots.get('mutable_slots'):
        await save({'status': 'no_slots', 'idx': idx}, results_list)
        print(f"[{idx}] no_slots (o3)")
        return

    n = len(slots['mutable_slots'])

    # Stage 2: o3 back-synthesis
    synth_text = await api_call(
        [{"role": "system", "content": BACK_SYNTHESIS},
         {"role": "user", "content": f"Original:\n{problem}\n\nSolution:\n{solution}\n\nSlots:\n{json.dumps(slots['mutable_slots'])}\n\nCreate variant."}],
        model="o3", max_tokens=6000)
    synth = extract_json(synth_text) if synth_text else None
    if not synth or not synth.get('new_problem'):
        await save({'status': 'error', 'idx': idx, 'reason': 'synthesis failed'}, results_list)
        print(f"[{idx}] synth_error")
        return

    new_p, new_s, new_a = synth['new_problem'], synth['new_solution'], synth.get('new_answer', '')

    # Stage 3: 3 judges + 3 repair rounds
    for rr in range(4):
        judges = await asyncio.gather(*[api_call(
            [{"role": "system", "content": VERIFY},
             {"role": "user", "content": f"Problem:\n{new_p}\n\nSolution:\n{new_s}"}],
            model="o3", max_tokens=500) for _ in range(3)])
        accepts = sum(1 for j in judges if j and 'ACCEPT' in j.upper() and 'REJECT' not in j.upper())
        if accepts == 3:
            await save({
                'status': 'accepted', 'idx': idx,
                'original_problem': problem, 'original_solution': solution,
                'kv_problem': new_p, 'kv_solution': new_s, 'kv_answer': new_a,
                'mutable_slots': slots['mutable_slots'],
                'slot_changes': synth.get('slot_changes', []),
                'repair_rounds': rr, 'n_slots': n,
            }, results_list)
            print(f"[{idx}] ACCEPTED (round {rr}, {n} slots)")
            return
        if rr < 3:
            reasons = [re.search(r'REASON:\s*(.*)', j or '', re.I) for j in judges]
            reason_str = '; '.join(m.group(1)[:200] for m in reasons if m)[:500]
            fix = await api_call(
                [{"role": "system", "content": REPAIR.format(problem=new_p, solution=new_s, reason=reason_str)},
                 {"role": "user", "content": "Fix."}],
                model="o3", max_tokens=6000)
            fd = extract_json(fix) if fix else None
            if fd:
                new_p = fd.get('new_problem', new_p)
                new_s = fd.get('new_solution', new_s)
                new_a = fd.get('new_answer', new_a)

    await save({'status': 'rejected', 'idx': idx}, results_list)
    print(f"[{idx}] REJECTED")

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

async def evaluate_all(all_accepted):
    async def solve(problem, model):
        async with SEM_EVAL:
            r = await client.chat.completions.create(
                model=model, temperature=0, max_tokens=2048,
                messages=[{"role": "system", "content": "Solve step by step. Final answer in \\boxed{}."},
                          {"role": "user", "content": problem}])
            return r.choices[0].message.content

    async def grade(ref, stu):
        async with SEM_EVAL:
            r = await client.chat.completions.create(
                model="gpt-4o", temperature=0, max_tokens=10,
                messages=[{"role": "user", "content": f"Are these equivalent? Ref: {ref}\nStudent: {stu}\nCORRECT or INCORRECT."}])
            t = r.choices[0].message.content.upper()
            return 'INCORRECT' not in t and 'CORRECT' in t

    results = {}
    for model in ['gpt-4o', 'gpt-4o-mini']:
        print(f"\nEval {len(all_accepted)} with {model}...")
        orig_sols = await asyncio.gather(*[solve(a['original_problem'], model) for a in all_accepted])
        kv_sols = await asyncio.gather(*[solve(a['kv_problem'], model) for a in all_accepted])

        og, kg = [], []
        for i, a in enumerate(all_accepted):
            ro = extract_boxed(a['original_solution']); so = extract_boxed(orig_sols[i])
            rk = a.get('kv_answer') or extract_boxed(a.get('kv_solution','')); sk = extract_boxed(kv_sols[i])
            og.append(await grade(ro or 'N/A', so or 'N/A') if ro and so else False)
            kg.append(await grade(rk or 'N/A', sk or 'N/A') if rk and sk else False)

        oa = sum(og)/len(og)*100; ka = sum(kg)/len(kg)*100
        results[model] = {'orig': oa, 'kv': ka, 'delta': ka-oa, 'n': len(all_accepted),
                          'orig_c': sum(og), 'kv_c': sum(kg)}
        print(f"  {model}: orig={oa:.1f}% kv={ka:.1f}% Δ={ka-oa:+.1f}pp (n={len(all_accepted)})")
    return results

async def main():
    # Load all Level 5 problems (same seed as kv_math_200.py)
    subsets = ['algebra', 'number_theory', 'precalculus', 'intermediate_algebra', 'counting_and_probability', 'geometry']
    all_l5 = []
    for s in subsets:
        ds = load_dataset('EleutherAI/hendrycks_math', s, split='test')
        for item in ds:
            if item.get('level') == 'Level 5' and len(item.get('solution','')) > 50:
                item['subject'] = s; all_l5.append(item)
    random.shuffle(all_l5)
    selected = all_l5[:200]
    print(f"Total pool: {len(selected)} Level 5 problems")

    # Load previous kv_200 results to find no_slots indices
    with open(os.path.join(OUTPUT_DIR, 'kv_generation.json')) as f:
        prev = json.load(f)
    no_slots_indices = [r['original_index'] for r in prev if r['status'] == 'no_slots']
    prev_accepted = [r for r in prev if r['status'] == 'accepted']
    print(f"Previous: {len(prev_accepted)} accepted, {len(no_slots_indices)} no_slots to redo with o3")

    # Also load kv_50 accepted
    kv50_file = '/home/yurenh2/gap/mini_gap_math_results/kv_50/kv_final_results.json'
    kv50_accepted = []
    if os.path.exists(kv50_file):
        with open(kv50_file) as f:
            kv50 = json.load(f)
        kv50_accepted = kv50.get('accepted_variants', [])
        print(f"kv_50 accepted: {len(kv50_accepted)}")

    # Resume redo progress
    redo_results = []
    done_indices = set()
    if os.path.exists(REDO_FILE):
        with open(REDO_FILE) as f:
            redo_results = json.load(f)
        done_indices = {r['idx'] for r in redo_results}
        print(f"Resuming redo: {len(redo_results)} done")

    remaining = [i for i in no_slots_indices if i not in done_indices]
    print(f"Remaining to redo: {len(remaining)}")

    # Process in batches of 8
    for batch_start in range(0, len(remaining), 8):
        batch = remaining[batch_start:batch_start+8]
        tasks = [process_one(selected[i]['problem'], selected[i]['solution'], i, redo_results) for i in batch]
        await asyncio.gather(*tasks)
        from collections import Counter
        st = Counter(r['status'] for r in redo_results)
        print(f"--- Redo progress: {len(redo_results)}/{len(no_slots_indices)}, {dict(st)} ---")

    # Combine all accepted
    redo_accepted = [r for r in redo_results if r['status'] == 'accepted']
    all_accepted = kv50_accepted + prev_accepted + redo_accepted
    print(f"\nTotal accepted: {len(all_accepted)} (kv50={len(kv50_accepted)}, kv200={len(prev_accepted)}, redo={len(redo_accepted)})")

    # Evaluate
    if all_accepted:
        eval_results = await evaluate_all(all_accepted)
        final = {
            'total_accepted': len(all_accepted),
            'sources': {'kv50': len(kv50_accepted), 'kv200': len(prev_accepted), 'redo': len(redo_accepted)},
            'evaluation': eval_results,
        }
        with open(os.path.join(OUTPUT_DIR, 'kv_combined_final.json'), 'w') as f:
            json.dump(final, f, indent=2)
        print(f"\n{'='*60}\nFINAL COMBINED RESULTS ({len(all_accepted)} KV variants)\n{'='*60}")
        for m, r in eval_results.items():
            print(f"  {m}: orig={r['orig']:.1f}% kv={r['kv']:.1f}% Δ={r['delta']:+.1f}pp (n={r['n']})")

asyncio.run(main())
