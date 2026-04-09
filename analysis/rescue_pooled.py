"""Pooled rescue analysis for the rebuttal headline.

Reports:
1. Per-variant pooled rebound rates with Wilson 95% CI for each condition
2. Pooled McNemar (paired) tests across all 4 models per variant
3. Pooled McNemar across all 5 surface variants for each model
4. Headline single-cell numbers
"""
from __future__ import annotations
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PATH = Path("/home/yurenh2/gap/analysis/rescue_results/rescue_30.jsonl")
OUT_PATH = Path("/home/yurenh2/gap/analysis/rescue_pooled_summary.json")


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2 * cum)


def main():
    rows = [json.loads(l) for l in open(PATH)]
    print(f"Loaded {len(rows)} rows\n")

    # case_grades[(model, variant, index)] = {cond: grade}
    case_grades = defaultdict(dict)
    for r in rows:
        case_grades[(r["model"], r["variant"], r["index"])][r["condition"]] = r.get("grade")

    variants_order = ["descriptive_long", "descriptive_long_confusing",
                      "descriptive_long_misleading", "garbled_string", "kernel_variant"]
    short = {"descriptive_long":"DL","descriptive_long_confusing":"DLC",
             "descriptive_long_misleading":"DLM","garbled_string":"GS","kernel_variant":"KV"}

    summary = {}

    print("=" * 92)
    print("HEADLINE: Rescue rebound by variant (pooled across 4 models)")
    print("=" * 92)
    print(f"{'Variant':<6} {'Condition':<14} {'k/n':>10} {'rate':>7} "
          f"{'95% Wilson CI':>20} {'Δ vs null':>11}")
    print("-" * 80)
    var_summary = {}
    for v in variants_order:
        # Pool counts across models
        cell_counts = defaultdict(lambda: {"k": 0, "n": 0})
        for k, grds in case_grades.items():
            if k[1] != v: continue
            for cond in ("null", "canonical_T2", "own_T2"):
                if cond in grds:
                    cell_counts[cond]["n"] += 1
                    if grds[cond] == "CORRECT":
                        cell_counts[cond]["k"] += 1
        # Wilson CIs
        per_cond = {}
        null_p = cell_counts["null"]["k"] / max(1, cell_counts["null"]["n"])
        for cond in ("null", "canonical_T2", "own_T2"):
            if cond not in cell_counts: continue
            c = cell_counts[cond]
            if c["n"] == 0: continue
            p, lo, hi = wilson_ci(c["k"], c["n"])
            delta = (p - null_p) * 100 if cond != "null" else 0.0
            per_cond[cond] = {"k": c["k"], "n": c["n"], "p": p, "ci": [lo, hi], "delta_pp": delta}
            print(f"{short[v]:<6} {cond:<14} {c['k']:>4}/{c['n']:>4}  "
                  f"{p*100:>5.1f}% [{lo*100:>5.1f}%, {hi*100:>5.1f}%] "
                  f"{'+' if delta > 0 else ('' if delta == 0 else '-')}{abs(delta):>5.1f} pp")
        # Pooled McNemar (own vs null, can vs null, own vs can)
        mc = {}
        for a, b in [("canonical_T2", "null"), ("own_T2", "null"),
                     ("own_T2", "canonical_T2")]:
            b_count = c_count = 0
            for k, grds in case_grades.items():
                if k[1] != v: continue
                ga = grds.get(a); gb = grds.get(b)
                if ga is None or gb is None: continue
                if ga == "CORRECT" and gb == "INCORRECT": b_count += 1
                elif ga == "INCORRECT" and gb == "CORRECT": c_count += 1
            p = mcnemar_p(b_count, c_count)
            mc[f"{a}_vs_{b}"] = {"b": b_count, "c": c_count, "p": p}
        var_summary[v] = {"per_cond": per_cond, "mcnemar": mc}
        print()

    summary["per_variant"] = var_summary

    # Pooled McNemar across all surface variants for canonical vs null and own vs null
    print("\n" + "=" * 92)
    print("POOLED McNEMAR (across all 4 surface variants × 4 models)")
    print("=" * 92)
    surface_vs = ["descriptive_long", "descriptive_long_confusing",
                  "descriptive_long_misleading", "garbled_string"]
    for a, b in [("canonical_T2", "null"), ("own_T2", "null"),
                 ("own_T2", "canonical_T2")]:
        b_count = c_count = 0
        for k, grds in case_grades.items():
            if k[1] not in surface_vs: continue
            ga = grds.get(a); gb = grds.get(b)
            if ga is None or gb is None: continue
            if ga == "CORRECT" and gb == "INCORRECT": b_count += 1
            elif ga == "INCORRECT" and gb == "CORRECT": c_count += 1
        p = mcnemar_p(b_count, c_count)
        n = b_count + c_count
        odds_ratio = b_count / max(1, c_count)
        print(f"  {a:<14} > {b:<14}  b={b_count:>4}, c={c_count:>4}  "
              f"OR={odds_ratio:>4.2f}  McNemar p={p:.2e}  (n_discordant={n})")
    # KV separately
    print()
    for a, b in [("canonical_T2", "null")]:
        b_count = c_count = 0
        for k, grds in case_grades.items():
            if k[1] != "kernel_variant": continue
            ga = grds.get(a); gb = grds.get(b)
            if ga is None or gb is None: continue
            if ga == "CORRECT" and gb == "INCORRECT": b_count += 1
            elif ga == "INCORRECT" and gb == "CORRECT": c_count += 1
        p = mcnemar_p(b_count, c_count)
        odds_ratio = b_count / max(1, c_count)
        print(f"  KV: {a:<14} > {b:<14}  b={b_count:>4}, c={c_count:>4}  "
              f"OR={odds_ratio:>4.2f}  McNemar p={p:.2e}")

    # Per model summary
    print("\n" + "=" * 92)
    print("PER MODEL (averaged across 4 surface variants)")
    print("=" * 92)
    print(f"{'Model':<22} {'null':>10} {'canonical_T2':>14} {'own_T2':>10} "
          f"{'can-null':>10} {'own-null':>10}")
    per_model = {}
    for model in sorted({k[0] for k in case_grades}):
        cnts = defaultdict(lambda: {"k": 0, "n": 0})
        for k, grds in case_grades.items():
            if k[0] != model: continue
            if k[1] not in surface_vs: continue
            for cond in ("null", "canonical_T2", "own_T2"):
                if cond in grds:
                    cnts[cond]["n"] += 1
                    if grds[cond] == "CORRECT":
                        cnts[cond]["k"] += 1
        nul_p = cnts["null"]["k"] / max(1, cnts["null"]["n"])
        can_p = cnts["canonical_T2"]["k"] / max(1, cnts["canonical_T2"]["n"])
        own_p = cnts["own_T2"]["k"] / max(1, cnts["own_T2"]["n"])
        per_model[model] = {
            "null": {"k": cnts["null"]["k"], "n": cnts["null"]["n"], "p": nul_p},
            "canonical_T2": {"k": cnts["canonical_T2"]["k"], "n": cnts["canonical_T2"]["n"], "p": can_p},
            "own_T2": {"k": cnts["own_T2"]["k"], "n": cnts["own_T2"]["n"], "p": own_p},
            "can_minus_null_pp": (can_p - nul_p) * 100,
            "own_minus_null_pp": (own_p - nul_p) * 100,
        }
        print(f"  {model:<20} {nul_p*100:>9.1f}% {can_p*100:>13.1f}% {own_p*100:>9.1f}% "
              f"{(can_p-nul_p)*100:>+9.1f}pp {(own_p-nul_p)*100:>+9.1f}pp")
    summary["per_model"] = per_model

    json.dump(summary, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
