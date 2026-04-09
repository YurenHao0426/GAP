#!/usr/bin/env python3
"""
calibrate_to_o3.py – End-to-end pipeline that
1. ingests existing o4-mini grading results for multiple models,
2. draws a budget-constrained stratified sample,
3. (optionally) re-grades those samples with o3 to obtain gold labels,
4. learns per-stratum error rates and calibrates all o4 labels to the o3 scale,
5. outputs required artefacts:
   – sample_list.csv
   – o3_raw.parquet (only when --run-o3)
   – calibrated_o3_scores.csv

Run:
    python calibrate_to_o3.py                # stop after sampling only
    python calibrate_to_o3.py --run-o3       # also call o3 re-grader

"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# Third-party library used by --run-o3 mode
try:
    from loader.openai_client import OpenAIModelLoader  # type: ignore
except ModuleNotFoundError:
    OpenAIModelLoader = None  # graceful degradation when running sampling-only mode

###############################################################################
# Constants – adjust here if the budget or cost model ever changes
###############################################################################
COST_PER_RECORD = 0.154  # USD per o3 grading request
BUDGET_MAX = 800.0       # USD hard cap
N_MAX = math.floor(BUDGET_MAX / COST_PER_RECORD)  # 5194 with default params
SEED = 42
MIN_PER_LAYER = 10

############################# Logging setup ###################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("calibrate")

###############################################################################
# Utility functions
###############################################################################

def wilson_ci(k: float, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a proportion.
    k may be fractional (calibrated successes). Returns (low, high)."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
        / denom
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def parse_diff(index: str) -> int:
    """Extract trailing difficulty digit (1-6) from an index like 2024-B-6."""
    try:
        return int(index.split("-")[-1])
    except (ValueError, IndexError):
        return -1  # fallback – will be filtered out later

###############################################################################
# 1 Load meta-data from dataset/*.json – mapping from problem index to (type,diff)
###############################################################################

def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Tuple[str, int]]:
    mapping: Dict[str, Tuple[str, int]] = {}
    json_files = sorted(dataset_dir.glob("*.json"))
    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            idx = data.get("index")
            typ = data.get("type")
            diff = parse_diff(idx)
            if idx and typ and diff != -1:
                mapping[idx] = (typ, diff)
        except Exception as e:
            LOGGER.warning(f"Failed to parse {fp}: {e}")
    LOGGER.info(f"Loaded metadata for {len(mapping):,} problems from dataset")
    return mapping

###############################################################################
# 2 Load all o4-mini result JSONs into one DataFrame
###############################################################################

def load_o4_results(results_root: Path, meta: Dict[str, Tuple[str, int]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    model_dirs = [d for d in results_root.iterdir() if d.is_dir()]
    for model_dir in model_dirs:
        model_id = model_dir.name
        # consider only *_original.json for uniformity
        for fp in model_dir.glob("*original.json"):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    res = json.load(f)
                for pr in res.get("problems", []):
                    idx = pr.get("index")
                    grade_info = pr.get("grade", {})
                    o4_score = int(grade_info.get("grade") == "CORRECT")
                    # meta info
                    typ, diff = meta.get(idx, (None, None))
                    if typ is None:
                        continue  # skip problems without meta
                    row = {
                        "id": idx,
                        "model_id": model_id,
                        "type": typ,
                        "diff": diff,
                        "o4_score": o4_score,
                        # Extra fields useful for optional o3 grading
                        "student_solution": pr.get("solve", {}).get("solution", ""),
                    }
                    rows.append(row)
            except Exception as e:
                LOGGER.warning(f"Failed to process {fp}: {e}")
    df = pd.DataFrame(rows)
    LOGGER.info(f"Ingested {len(df):,} problem-model pairs across {df['model_id'].nunique()} models")
    return df

###############################################################################
# 3 Stratified sampling under budget
###############################################################################

def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    group_cols = ["type", "diff", "o4_score"]

    # Compute desired sample sizes per layer
    layer_counts = df.groupby(group_cols, observed=True).size().rename("N_k")
    total_records = len(df)
    target_sizes = (
        (layer_counts / total_records * N_MAX).apply(np.ceil).astype(int).clip(lower=MIN_PER_LAYER)
    )

    # If the initial allocation exceeds budget, scale down proportionally (but keep >=MIN_PER_LAYER)
    total_target = target_sizes.sum()
    if total_target > N_MAX:
        LOGGER.info(
            f"Initial allocation {total_target} exceeds N_MAX={N_MAX}. Scaling down proportionally."
        )
        scaling = (N_MAX - MIN_PER_LAYER * target_sizes.size) / (
            total_target - MIN_PER_LAYER * target_sizes.size
        )
        scaling = max(scaling, 0.0)
        target_sizes = (
            MIN_PER_LAYER
            + np.floor((target_sizes - MIN_PER_LAYER) * scaling).astype(int)
        )
    LOGGER.info(
        f"Final per-stratum sample sizes prepared (sum={target_sizes.sum()}) – within budget"
    )

    # Actual sampling
    samples = []
    for key, group in df.groupby(group_cols, observed=True):
        n = min(target_sizes.get(key, MIN_PER_LAYER), len(group))
        if n <= 0:
            continue
        sample_idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        samples.append(df.loc[sample_idx])
    sample_df = pd.concat(samples, ignore_index=True)
    LOGGER.info(f"Sampled {len(sample_df):,} rows in total (<= {N_MAX})")
    return sample_df

###############################################################################
# 4 Async o3 re-grading helper
###############################################################################

async def grade_with_o3(sample_df: pd.DataFrame, meta: Dict[str, Tuple[str, int]]) -> pd.Series:
    """Returns pd.Series of int o3_score aligned with sample_df.index."""
    if OpenAIModelLoader is None:
        raise RuntimeError("OpenAIModelLoader not available. Install dependencies or run without --run-o3.")

    async with OpenAIModelLoader(solver_model="o3", grader_model="o3") as loader:

        async def grade_one(row) -> int:
            idx = row.id
            question = None
            reference_solution = None
            # load dataset file lazily when needed
            dataset_file = Path("dataset") / f"{idx}.json"
            if dataset_file.exists():
                try:
                    with dataset_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    question = data.get("question", "")
                    reference_solution = data.get("solution", "")
                except Exception:
                    pass
            if not question:
                return -1  # cannot grade
            student_solution = row.student_solution or ""
            try:
                grade_result, _ = await loader.grade_solution(
                    question,
                    student_solution,
                    reference_solution,
                    problem_type="proof",
                    model="o3",
                )
                return int(grade_result.get("grade") == "CORRECT") if grade_result else -1
            except Exception as exc:
                LOGGER.warning(f"o3 grading failed for {idx}: {exc}")
                return -1

        sem = asyncio.Semaphore(20)
        async def sem_grade(row):
            async with sem:
                return await grade_one(row)

        tasks = [asyncio.create_task(sem_grade(row)) for _, row in sample_df.iterrows()]
        o3_scores = await asyncio.gather(*tasks)
        return pd.Series(o3_scores, index=sample_df.index, name="o3_score")

###############################################################################
# 5 Calibration – compute per-stratum error rates and apply
###############################################################################

def compute_error_rates(sample_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["type", "diff"]

    # Build contingency counts per stratum
    counts = sample_df.groupby(group_cols + ["o4_score", "o3_score"], observed=True).size().unstack(fill_value=0)
    # Ensure o3_score columns 0 and 1 exist
    for col in [0, 1]:
        if col not in counts.columns:
            counts[col] = 0
    # counts index columns: type, diff, o4_score
    # Compute p1_k and p0_k
    records = []
    for (typ, diff, o4_val), row in counts.reset_index().groupby(["type", "diff", "o4_score"], observed=True):
        n = row[[0, 1]].sum(axis=1).values[0]
        k = row[0].values[0]  # for p1 or p0 depends
        if o4_val == 1:  # looking at false positives (o4=1 but o3=0)
            p1 = k / n if n else 0.10
            records.append({"type": typ, "diff": diff, "p1": p1})
        else:  # o4=0
            p0 = row[1].values[0] / n if n else 0.10
            records.append({"type": typ, "diff": diff, "p0": p0})
    errs = pd.DataFrame(records).groupby(["type", "diff"], observed=True).first().reset_index()
    errs["p1"].fillna(0.10, inplace=True)
    errs["p0"].fillna(0.10, inplace=True)
    return errs


def apply_calibration(full_df: pd.DataFrame, err_df: pd.DataFrame) -> pd.Series:
    merged = full_df.merge(err_df, on=["type", "diff"], how="left")
    merged["p1"].fillna(0.10, inplace=True)
    merged["p0"].fillna(0.10, inplace=True)
    est = np.where(merged.o4_score == 1, 1 - merged.p1, merged.p0)
    return pd.Series(est, index=full_df.index, name="o3_est")

###############################################################################
# 6 Main entry
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Calibrate o4-mini results to o3 scale")
    parser.add_argument("--run-o3", action="store_true", help="Actually call o3 to grade the sampled pairs")
    parser.add_argument("--output-dir", default="calibration_out", help="Directory to store generated artefacts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 Load meta and results
    meta = load_dataset_metadata(Path("dataset"))
    full_df = load_o4_results(Path("results"), meta)

    # 2 Sampling
    sample_df = stratified_sample(full_df)
    sample_df.to_csv(out_dir / "sample_list.csv", index=False)

    if args.run_o3:
        LOGGER.info("Starting o3 re-grading – this may incur cost!")
        start = asyncio.run(grade_with_o3(sample_df, meta))
        sample_df["o3_score"] = start
        sample_df.to_parquet(out_dir / "o3_raw.parquet", index=False)
        spent = sample_df["o3_score"].notna().sum() * COST_PER_RECORD
        LOGGER.info(f"o3 grading finished. Cost ≈ ${spent:.2f}")
    else:
        LOGGER.info("--run-o3 not provided; skipping API calls and downstream calibration")
        return  # exit early

    # 3 Calibration
    err_df = compute_error_rates(sample_df)
    full_df["o3_est"] = apply_calibration(full_df, err_df)

    # 4 Aggregate per model
    agg_rows = []
    for model_id, grp in full_df.groupby("model_id", observed=True):
        mean_est = grp.o3_est.mean()
        n = len(grp)
        k_hat = mean_est * n
        ci_low, ci_high = wilson_ci(k_hat, n)
        agg_rows.append({
            "model_id": model_id,
            "mean": mean_est,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "calibrated_o3_scores.csv", index=False)
    LOGGER.info("Calibration finished. Artefacts saved to %s", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("Fatal error: %s", exc)
        raise 