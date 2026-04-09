"""End-to-end rescue experiment runner.

For each (model, variant, flip-case):
  - Build 3 prompts: own_T2, canonical_T2, null  (KV: only canonical_T2 + null)
  - Solve with the same model the case originally failed under
  - Grade with gpt-4o using the variant problem + canonical variant solution as reference
  - Save per-case results immediately to a jsonl checkpoint (resumable)

Usage:
  python rescue_runner.py --pilot   # 5 cases per cell (smoke test)
  python rescue_runner.py           # 30 cases per cell (full run)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Local imports
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from rescue_prompts import (
    truncate_T2, rename_own_prefix,
    build_rescue_prompt, build_null_prompt, NULL_SCAFFOLD,
)
from rescue_api import (
    SOLVER_PROVIDERS, solve, grade, parse_solution, parse_grade,
)
from structural_overlap import (
    DATASET_DIR, RESULTS_DIR, find_variant_file, load_problems, SURFACE_VARIANTS,
)


# Short model name -> directory name in results_new
MODEL_RESULTS_DIRS = {
    "gpt-4.1-mini":     "gpt-4.1-mini",
    "gpt-4o-mini":      "gpt-4o-mini",
    "claude-sonnet-4":  "claude-sonnet-4",
    "gemini-2.5-flash": "gemini_2.5_flash",  # historical underscore naming
}
SELECTED_MODELS = ["gpt-4.1-mini", "gpt-4o-mini", "claude-sonnet-4", "gemini-2.5-flash"]
ALL_VARIANTS = SURFACE_VARIANTS + ["kernel_variant"]
SURFACE_CONDITIONS = ["own_T2", "canonical_T2", "null"]
KV_CONDITIONS = ["canonical_T2", "null"]


# ---------- Dataset loading ----------

def load_dataset_full() -> dict:
    """Returns: {idx: {original: {...}, variants: {v: {map, question, solution}}}}.

    The dataset stores top-level question/solution and variant-keyed question/solution/map.
    """
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        cell = {
            "problem_type": d.get("problem_type"),
            "original_question": d.get("question") or "",
            "original_solution": d.get("solution") or "",
            "variants": {},
        }
        for v, vd in d.get("variants", {}).items():
            if isinstance(vd, dict):
                rmap = vd.get("map")
                if isinstance(rmap, str):
                    try:
                        rmap = eval(rmap, {"__builtins__": {}}, {})
                    except Exception:
                        rmap = None
                cell["variants"][v] = {
                    "question": vd.get("question") or "",
                    "solution": vd.get("solution") or "",
                    "map": rmap if isinstance(rmap, dict) else None,
                }
        out[idx] = cell
    return out


# ---------- Flip case selection ----------

def find_flip_cases(model: str, variant: str, max_cases: int,
                    seed: int = 42) -> list:
    """Identify (orig_correct, var_wrong) flip cases for the cell.

    Returns list of dicts with: index, problem_type, model_orig_solution,
    final_answer (recorded), variant_problem_statement (from results).
    """
    mdir = RESULTS_DIR / MODEL_RESULTS_DIRS.get(model, model)
    op = find_variant_file(mdir, "original")
    vp = find_variant_file(mdir, variant)
    if not op or not vp:
        return []
    orig_by = {p["index"]: p for p in load_problems(op)}
    var_by = {p["index"]: p for p in load_problems(vp)}
    cases = []
    for idx in sorted(set(orig_by) & set(var_by)):
        po, pv = orig_by[idx], var_by[idx]
        if po.get("correct") is not True or pv.get("correct") is not False:
            continue
        orig_text = (po.get("solve") or {}).get("solution") or ""
        if not orig_text:
            continue
        # Skip cases where we couldn't extract a T2 prefix from the original
        fa = (po.get("solve") or {}).get("final_answer") or ""
        if truncate_T2(orig_text, fa) is None:
            continue
        cases.append({
            "index": idx,
            "problem_type": po.get("problem_type"),
            "orig_solution": orig_text,
            "orig_final_answer": fa,
        })
    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases[:max_cases]


# ---------- Prompt construction per case ----------

def build_case_prompts(case: dict, variant: str, ds_cell: dict) -> dict:
    """Returns: {condition_name: user_message_string}."""
    var_info = ds_cell["variants"].get(variant, {})
    var_question = var_info.get("question", "")
    if not var_question:
        return {}
    prompts = {}
    is_kv = (variant == "kernel_variant")

    # canonical_T2: dataset's canonical variant solution truncated
    canon_sol = var_info.get("solution", "")
    if canon_sol:
        canon_pre = truncate_T2(canon_sol, None)
        if canon_pre:
            prompts["canonical_T2"] = build_rescue_prompt(var_question, canon_pre)

    # own_T2: only for surface variants — model's own original-correct prefix renamed
    if not is_kv:
        rmap = var_info.get("map") or {}
        own_pre = truncate_T2(case["orig_solution"], case.get("orig_final_answer"))
        if own_pre and rmap:
            renamed = rename_own_prefix(own_pre, rmap)
            prompts["own_T2"] = build_rescue_prompt(var_question, renamed)

    # null: always available
    prompts["null"] = build_null_prompt(var_question)
    return prompts


# ---------- Per-condition runner ----------

async def run_one_condition(model: str, condition: str, user_msg: str,
                             case: dict, variant: str, ds_cell: dict) -> dict:
    """Solve + grade a single condition for a single case. Returns a result dict."""
    var_info = ds_cell["variants"].get(variant, {})
    var_question = var_info.get("question", "")
    canon_sol = var_info.get("solution", "")
    problem_type = case["problem_type"]
    t0 = time.time()
    solve_resp = await solve(model, user_msg)
    solve_dt = time.time() - t0
    if solve_resp["status"] != "success":
        return {
            "model": model, "variant": variant, "condition": condition,
            "index": case["index"], "problem_type": problem_type,
            "solve_status": "failed",
            "solve_error": solve_resp["error"],
            "solve_seconds": solve_dt,
            "grade": None,
        }
    parsed = parse_solution(solve_resp["content"])
    if not parsed["solution"]:
        return {
            "model": model, "variant": variant, "condition": condition,
            "index": case["index"], "problem_type": problem_type,
            "solve_status": "parse_failed",
            "solve_error": parsed.get("_parse_error"),
            "solve_seconds": solve_dt,
            "raw_solve_content": solve_resp["content"][:500],
            "grade": None,
        }
    student_solution = parsed["solution"]
    t1 = time.time()
    grade_resp = await grade(problem_type, var_question, student_solution, canon_sol)
    grade_dt = time.time() - t1
    if grade_resp["status"] != "success":
        return {
            "model": model, "variant": variant, "condition": condition,
            "index": case["index"], "problem_type": problem_type,
            "solve_status": "success",
            "solve_seconds": solve_dt,
            "grade_seconds": grade_dt,
            "grade_status": "failed",
            "grade_error": grade_resp["error"],
            "student_solution_len": len(student_solution),
            "student_final_answer": parsed["final_answer"],
            "grade": None,
        }
    parsed_grade = parse_grade(grade_resp["content"])
    return {
        "model": model, "variant": variant, "condition": condition,
        "index": case["index"], "problem_type": problem_type,
        "solve_status": "success",
        "solve_seconds": solve_dt,
        "grade_seconds": grade_dt,
        "grade_status": "success",
        "student_solution_len": len(student_solution),
        "student_solution": student_solution,  # full text for downstream analysis
        "student_final_answer": parsed["final_answer"][:500],
        "grade": parsed_grade["grade"],
        "final_answer_correct": parsed_grade.get("final_answer_correct"),
        "grade_feedback": (parsed_grade.get("detailed_feedback") or "")[:1000],
    }


# ---------- Main run ----------

OUT_DIR = Path("/home/yurenh2/gap/analysis/rescue_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_keys(path: Path) -> set:
    """Read jsonl checkpoint and return set of (cell_key, condition, index)."""
    keys = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                keys.add((d["model"], d["variant"], d["condition"], d["index"]))
            except Exception:
                pass
    return keys


async def run_all(num_cases_per_cell: int, dry_run: bool = False, models=None,
                  variants=None):
    print(f"Loading dataset ...", flush=True)
    ds = load_dataset_full()
    print(f"  loaded {len(ds)} problems", flush=True)

    out_path = OUT_DIR / f"rescue_{num_cases_per_cell}.jsonl"
    existing = load_existing_keys(out_path)
    print(f"Output: {out_path}  (existing rows: {len(existing)})")

    models = models or SELECTED_MODELS
    variants = variants or ALL_VARIANTS

    # Build the full task list
    tasks_to_run = []
    cell_summary = {}
    for model in models:
        for variant in variants:
            cases = find_flip_cases(model, variant, num_cases_per_cell)
            cell_key = f"{model}/{variant}"
            cell_summary[cell_key] = {"flip_cases_found": len(cases),
                                       "added_tasks": 0}
            for case in cases:
                ds_cell = ds.get(case["index"])
                if ds_cell is None:
                    continue
                prompts = build_case_prompts(case, variant, ds_cell)
                for cond, user_msg in prompts.items():
                    key = (model, variant, cond, case["index"])
                    if key in existing:
                        continue
                    tasks_to_run.append((model, variant, cond, case, ds_cell, user_msg))
                    cell_summary[cell_key]["added_tasks"] += 1

    print(f"\nCell-level plan ({num_cases_per_cell} flip cases each):")
    for k, v in sorted(cell_summary.items()):
        print(f"  {k:<46}  found={v['flip_cases_found']:>3}  new_tasks={v['added_tasks']:>4}")
    total = len(tasks_to_run)
    print(f"\nTotal new tasks: {total}")
    if dry_run:
        return

    if not tasks_to_run:
        print("Nothing to do.")
        return

    # Execute concurrently. Use a writer task to drain results into the jsonl.
    fout = open(out_path, "a")
    write_lock = asyncio.Lock()
    completed = 0
    failed = 0
    started_at = time.time()

    async def run_and_write(model, variant, cond, case, ds_cell, user_msg):
        nonlocal completed, failed
        try:
            res = await run_one_condition(model, cond, user_msg, case, variant, ds_cell)
        except Exception as e:
            res = {
                "model": model, "variant": variant, "condition": cond,
                "index": case["index"], "problem_type": case.get("problem_type"),
                "solve_status": "exception",
                "solve_error": f"{type(e).__name__}: {str(e)[:300]}",
                "grade": None,
            }
            failed += 1
        async with write_lock:
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            completed += 1
            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed:>4}/{total}] elapsed={elapsed:>5.0f}s "
                      f"rate={rate:>4.1f}/s eta={eta:>5.0f}s "
                      f"failed_so_far={failed}", flush=True)

    awaitables = [run_and_write(*t) for t in tasks_to_run]
    await asyncio.gather(*awaitables)
    fout.close()
    print(f"\nDone. {completed}/{total} written.  Failed: {failed}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="run only 5 cases per cell")
    ap.add_argument("--cases", type=int, default=30, help="cases per cell (full run)")
    ap.add_argument("--dry-run", action="store_true", help="print plan, don't call APIs")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    args = ap.parse_args()
    n = 5 if args.pilot else args.cases
    asyncio.run(run_all(n, dry_run=args.dry_run,
                        models=args.models, variants=args.variants))


if __name__ == "__main__":
    main()
