#!/usr/bin/env python3
"""
Re-grade an existing results JSON file using a (possibly different) grader model.

The script loads a results file produced by `batch_evaluate.py` (or a compatible
JSON list) and re-grades every problem using the specified grader.  No solving
is performed – instead we reuse the previously generated solutions stored in
`solve.solution`.

Example usage
-------------
python regrade.py \
    --results-file results/o3/o3_original.json \
    --dataset-dir dataset/ \
    --provider openai \
    --grader-model o3 \
    --max-concurrent 5 \
    --output results/regraded_o3_original.json

"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import logging

# Determine directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # one level up

# Add both the script dir and project root to PYTHONPATH to locate 'loader'
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PROJECT_ROOT))

from loader import create_loader  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False

    class tqdm:  # type: ignore
        """Minimal fallback if tqdm is not available."""

        def __init__(self, total=None, desc=None, **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc or ""
            print(f"{self.desc}: starting …")

        def update(self, n=1):
            self.n += n
            if self.total:
                pct = self.n / self.total * 100
                print(f"{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end="\r")

        def set_postfix(self, _):
            pass

        def close(self):
            print()  # newline


###############################################################################
# Helper functions
###############################################################################


def load_dataset(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read every JSON file in *dataset_dir* and return a mapping index → data."""
    dataset: Dict[str, Dict[str, Any]] = {}
    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                idx = data.get("index")
                if idx:
                    dataset[idx] = data
        except Exception as exc:  # pragma: no cover – best-effort ingest
            logging.warning("Failed to load %s: %s", json_file, exc)
    return dataset


async def regrade_problem(loader,  # type: ignore[valid-type]
                          problem_record: Dict[str, Any],
                          dataset_entry: Dict[str, Any],
                          variant_type: str) -> Dict[str, Any]:
    """Re-grade one problem and return a new result dict."""

    idx = problem_record.get("index", "unknown")
    problem_type = dataset_entry.get("problem_type", "proof")

    # Extract question & reference solution according to variant
    if variant_type == "original":
        question = str(dataset_entry.get("question", "")).strip()
        reference_solution = str(dataset_entry.get("solution", "")).strip()
    else:
        variant = dataset_entry.get("variants", {}).get(variant_type, {})
        question = str(variant.get("question", "")).strip()
        reference_solution = str(variant.get("solution", "")).strip()

    if not question or not reference_solution:
        return {
            "index": idx,
            "status": "skipped",
            "reason": "missing_fields",
        }

    # Previously generated solution
    student_solution = str(problem_record.get("solve", {}).get("solution", "")).strip()
    final_answer = str(problem_record.get("solve", {}).get("final_answer", "")).strip()

    # Grade the solution (temperature hard-coded inside create_loader for o-series)
    grade_result, _raw = await loader.grade_solution(
        question,
        student_solution,
        reference_solution,
        problem_type,
    )

    # Build merged record retaining original fields + new grade
    new_record = {
        "index": idx,
        "variant_type": variant_type,
        "problem_type": problem_type,
        "solve": {
            "solution": student_solution,
            "final_answer": final_answer,
        },
        "grade": grade_result or {"status": "failed"},
    }

    # Convenience shortcut for correctness
    new_record["correct"] = new_record["grade"].get("grade") == "CORRECT"
    return new_record


###############################################################################
# Main orchestration
###############################################################################


async def main() -> None:  # noqa: C901 – single entry-point
    parser = argparse.ArgumentParser(description="Re-grade an existing results file")
    parser.add_argument("--results-file", required=True, type=Path, help="Path to existing results JSON")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Directory containing dataset JSON files")
    parser.add_argument("--provider", default="openai", help="Grader provider (default: openai)")
    parser.add_argument("--grader-model", default="o3", help="Grader model name (default: o3)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent API calls")
    parser.add_argument("--variant-type", default="original", help="Problem variant used in results file")
    parser.add_argument("--output", type=Path, help="Where to write re-graded results (JSON)")
    parser.add_argument("--quick", action="store_true", help="Quick mode – single retry, shorter timeouts")
    parser.add_argument("--debug", action="store_true", help="Verbose JSON-parsing debug")

    args = parser.parse_args()

    # Configure logging early
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.results_file.exists():
        logging.error("Results file %s does not exist", args.results_file)
        sys.exit(1)

    if not args.dataset_dir.exists():
        logging.error("Dataset directory %s does not exist", args.dataset_dir)
        sys.exit(1)

    # Load dataset into memory once
    logging.info("Loading dataset from %s", args.dataset_dir)
    dataset_map = load_dataset(args.dataset_dir)
    logging.info("Loaded %d dataset entries", len(dataset_map))

    # Load results JSON (support two formats: {'problems':[...]} or simple list)
    with open(args.results_file, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    if isinstance(raw_data, dict) and "problems" in raw_data:
        original_problems: List[Dict[str, Any]] = raw_data["problems"]  # type: ignore[assignment]
    elif isinstance(raw_data, list):
        original_problems = raw_data  # type: ignore[assignment]
    else:
        logging.error("Unsupported results file structure – expected list or dict with key 'problems'.")
        sys.exit(1)

    if not original_problems:
        logging.warning("No problems found in results file – nothing to re-grade.")
        sys.exit(0)

    # Create loader – we only need grader, but solver_model must be provided; reuse grader_model
    loader = create_loader(
        args.provider,
        solver_model=args.grader_model,
        grader_model=args.grader_model,
        quick=args.quick,
        debug=args.debug,
    )

    if not await loader.health_check():
        logging.error("Health check failed for provider %s", args.provider)
        sys.exit(1)

    # Estimate costs (rough – assumes avg lengths; tweak as needed)
    cost_info = await loader.estimate_cost(len(original_problems))
    logging.info("Estimated grading cost: $%.2f", cost_info.get("total_cost", 0))

    # Concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def wrapper(problem_record):
        idx = problem_record.get("index", "unknown")
        if idx not in dataset_map:
            logging.warning("Dataset entry for index %s not found – skipping", idx)
            return {"index": idx, "status": "skipped", "reason": "dataset_missing"}
        async with semaphore:
            return await regrade_problem(
                loader,
                problem_record,
                dataset_map[idx],
                args.variant_type,
            )

    # Progress bar setup
    pbar = tqdm(total=len(original_problems), desc="Re-grading")
    results: List[Dict[str, Any]] = []

    async def gather_tasks():
        for coro in asyncio.as_completed([wrapper(rec) for rec in original_problems]):
            res = await coro
            results.append(res)
            pbar.update(1)
    await gather_tasks()
    pbar.close()

    # Build summary
    completed = [r for r in results if r.get("grade", {}).get("status") == "success"]
    grades = [r["grade"].get("grade") for r in completed]
    numeric = [5.0 if g == "CORRECT" else 2.5 for g in grades]

    summary = {
        "total_problems": len(results),
        "completed": len(completed),
        "correct": sum(1 for g in grades if g == "CORRECT"),
        "incorrect": sum(1 for g in grades if g == "INCORRECT"),
        "average_grade": sum(numeric) / len(numeric) if numeric else 0.0,
        "provider": args.provider,
        "grader_model": args.grader_model,
        "variant_type": args.variant_type,
        "estimated_cost": cost_info,
        "timestamp": datetime.now().isoformat(),
    }

    output_payload = {
        "summary": summary,
        "problems": results,
    }

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        stem = args.results_file.stem + f"_regraded_{args.grader_model}"
        out_path = args.results_file.with_name(stem + args.results_file.suffix)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, ensure_ascii=False)
    logging.info("Saved re-graded results to %s", out_path)

    # Clean up HTTP client if applicable
    if hasattr(loader, "__aexit__"):
        await loader.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main()) 