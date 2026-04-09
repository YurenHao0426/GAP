"""Quantify spontaneous variant->canonical name normalization in own_T2 outputs.

For each own_T2 case, check whether the model's student_solution preserves the
variant variable names from its prefix or normalizes them back to the canonical
names from the dataset's rename map.

For each variant variable name in the rename map:
- count its occurrences in the prefix (as injected)
- count its occurrences in the model's student_solution
- count occurrences of the corresponding CANONICAL name in the student_solution

If the model preserves variant naming: variant_name count in solution should be
proportionally similar to the prefix count.
If the model normalizes back: canonical_name count in solution should rise while
variant_name count drops.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from rescue_runner import load_dataset_full, find_flip_cases, build_case_prompts

PILOT_PATH = Path("/home/yurenh2/gap/analysis/rescue_results/rescue_5.jsonl")


def count_word(text: str, word: str) -> int:
    """Whole-word count of `word` in `text`."""
    if not text or not word:
        return 0
    pat = r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])"
    return len(re.findall(pat, text))


def analyze_one(row: dict, ds_cell: dict) -> dict:
    """For one own_T2 row, compute name preservation stats."""
    variant = row["variant"]
    var_info = ds_cell["variants"].get(variant, {})
    rmap = var_info.get("map") or {}
    if not rmap:
        return {}
    student = row.get("student_solution") or ""
    if not student:
        return {}

    # Build the prefix that the model was given
    case = {
        "index": row["index"],
        "problem_type": row["problem_type"],
        "orig_solution": "",
        "orig_final_answer": "",
    }
    # We need the original solution from results_new to reconstruct the prefix.
    # Use find_flip_cases to recover it cleanly.
    cases = find_flip_cases(row["model"], variant, 100)
    matched = next((c for c in cases if c["index"] == row["index"]), None)
    if matched is None:
        return {}
    prompts = build_case_prompts(matched, variant, ds_cell)
    own_prompt = prompts.get("own_T2", "")
    if "PARTIAL WORK" not in own_prompt:
        return {}
    # Extract just the partial work text
    section = own_prompt.split("PARTIAL WORK")[1].split("Provide a complete")[0]
    section = section.split("(to copy verbatim")[1] if "(to copy verbatim" in section else section
    section = section.split("):", 1)[1] if "):" in section else section
    prefix = section.strip()

    # For each variant variable, count occurrences in prefix and in student
    per_var = {}
    for canon_name, var_name in rmap.items():
        if not var_name:
            continue
        prefix_v = count_word(prefix, var_name)
        student_v = count_word(student, var_name)
        student_c = count_word(student, canon_name)
        # Only meaningful if the variant name actually appeared in the prefix
        if prefix_v == 0:
            continue
        per_var[var_name] = {
            "canon_name": canon_name,
            "prefix_count_variant": prefix_v,
            "student_count_variant": student_v,
            "student_count_canonical": student_c,
            # Preservation ratio: how much of the variant naming survived
            # capped to 1.0 (model may use the variable many more times in
            # its continuation, which inflates the count)
            "preservation_ratio": min(1.0, student_v / max(1, prefix_v)),
            "normalization_ratio": min(1.0, student_c / max(1, prefix_v)),
        }
    if not per_var:
        return {}
    # Aggregate per case: median preservation
    pres_vals = [v["preservation_ratio"] for v in per_var.values()]
    norm_vals = [v["normalization_ratio"] for v in per_var.values()]
    return {
        "model": row["model"],
        "variant": variant,
        "index": row["index"],
        "grade": row.get("grade"),
        "n_vars_in_prefix": len(per_var),
        "median_preservation": statistics.median(pres_vals),
        "median_normalization": statistics.median(norm_vals),
        "mean_preservation": statistics.fmean(pres_vals),
        "mean_normalization": statistics.fmean(norm_vals),
        "per_var": per_var,
    }


def main():
    print("Loading dataset ...")
    ds = load_dataset_full()
    print(f"Loaded {len(ds)} problems")
    print(f"\nLoading pilot rows from {PILOT_PATH} ...")
    rows = [json.loads(l) for l in open(PILOT_PATH)]
    own_rows = [r for r in rows if r["condition"] == "own_T2"]
    print(f"  total rows: {len(rows)}, own_T2 rows: {len(own_rows)}")

    analyses = []
    skipped = 0
    for r in own_rows:
        ds_cell = ds.get(r["index"])
        if ds_cell is None:
            skipped += 1
            continue
        a = analyze_one(r, ds_cell)
        if a:
            analyses.append(a)
        else:
            skipped += 1
    print(f"  analyzed: {len(analyses)}, skipped: {skipped}")

    # Aggregate by variant
    print("\n=== SPONTANEOUS NORMALIZATION (own_T2 condition only) ===\n")
    print("Per case: median across variant variables of preservation ratio")
    print("(higher = more variant naming preserved; lower = normalized back to canonical)")
    print()
    print(f"{'Variant':<32} {'n':>4} {'median_pres':>12} {'mean_pres':>10} "
          f"{'median_norm':>12} {'mean_norm':>10}")
    print("-" * 90)
    by_variant = defaultdict(list)
    for a in analyses:
        by_variant[a["variant"]].append(a)
    for v in sorted(by_variant):
        cs = by_variant[v]
        mp_vals = [c["median_preservation"] for c in cs]
        mn_vals = [c["median_normalization"] for c in cs]
        print(f"{v:<32} {len(cs):>4} "
              f"{statistics.median(mp_vals):>12.3f} {statistics.fmean(mp_vals):>10.3f} "
              f"{statistics.median(mn_vals):>12.3f} {statistics.fmean(mn_vals):>10.3f}")

    # Aggregate by model
    print(f"\n{'Model':<22} {'n':>4} {'median_pres':>12} {'mean_pres':>10} "
          f"{'median_norm':>12} {'mean_norm':>10}")
    print("-" * 80)
    by_model = defaultdict(list)
    for a in analyses:
        by_model[a["model"]].append(a)
    for m in sorted(by_model):
        cs = by_model[m]
        mp_vals = [c["median_preservation"] for c in cs]
        mn_vals = [c["median_normalization"] for c in cs]
        print(f"{m:<22} {len(cs):>4} "
              f"{statistics.median(mp_vals):>12.3f} {statistics.fmean(mp_vals):>10.3f} "
              f"{statistics.median(mn_vals):>12.3f} {statistics.fmean(mn_vals):>10.3f}")

    # Effect of normalization on rebound: do cases that normalized more often FAIL?
    print("\n=== RELATION TO REBOUND ===")
    pass_pres = [a["median_preservation"] for a in analyses if a["grade"] == "CORRECT"]
    fail_pres = [a["median_preservation"] for a in analyses if a["grade"] == "INCORRECT"]
    print(f"  median_preservation among rebound CORRECT  (n={len(pass_pres)}): "
          f"median={statistics.median(pass_pres):.3f}  mean={statistics.fmean(pass_pres):.3f}")
    print(f"  median_preservation among rebound INCORRECT (n={len(fail_pres)}): "
          f"median={statistics.median(fail_pres):.3f}  mean={statistics.fmean(fail_pres):.3f}")

    # Save detailed results
    out = Path("/home/yurenh2/gap/analysis/normalization_results.json")
    json.dump([{k: v for k, v in a.items() if k != "per_var"} for a in analyses],
              open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
