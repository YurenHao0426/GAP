"""Analyze full rescue results: per-cell rebound rates, Wilson CIs, McNemar."""
from __future__ import annotations
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PATH = Path("/home/yurenh2/gap/analysis/rescue_results/rescue_30.jsonl")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_p(b: int, c: int) -> float:
    """McNemar exact-ish p (binomial two-sided). b = treat A correct, B wrong;
    c = treat A wrong, B correct. Returns p value testing b == c."""
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided binomial test on min(b,c) ~ Bin(n, 0.5)
    k = min(b, c)
    # cumulative
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * cum)
    return p


def main():
    rows = [json.loads(l) for l in open(PATH)]
    print(f"Loaded {len(rows)} rows")

    # Quick sanity
    from collections import Counter
    print("Solve status:", Counter(r.get("solve_status") for r in rows))
    print("Grade status:", Counter(r.get("grade_status") for r in rows))

    # Per-cell counts
    counts = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        if r.get("grade_status") != "success" and r.get("grade") not in ("CORRECT", "INCORRECT"):
            # Treat solve failures / parse failures as INCORRECT (conservative)
            pass
        key = (r["model"], r["variant"], r["condition"])
        counts[key]["total"] += 1
        if r.get("grade") == "CORRECT":
            counts[key]["correct"] += 1

    # Aggregated by (variant, condition)
    by_var_cond = defaultdict(lambda: {"total": 0, "correct": 0})
    for (m, v, c), d in counts.items():
        by_var_cond[(v, c)]["total"] += d["total"]
        by_var_cond[(v, c)]["correct"] += d["correct"]

    print("\n" + "=" * 90)
    print("REBOUND RATE BY (VARIANT, CONDITION) [aggregated across 4 models]")
    print("=" * 90)
    print(f"{'Variant':<32} {'Condition':<14} {'k/n':>10} {'rate':>7} {'95% Wilson CI':>20}")
    print("-" * 90)
    variants_order = ["descriptive_long", "descriptive_long_confusing",
                      "descriptive_long_misleading", "garbled_string", "kernel_variant"]
    conds_order = ["null", "canonical_T2", "own_T2"]
    for v in variants_order:
        for c in conds_order:
            d = by_var_cond.get((v, c))
            if not d:
                continue
            p, lo, hi = wilson_ci(d["correct"], d["total"])
            print(f"{v:<32} {c:<14} {d['correct']:>4}/{d['total']:>4}  "
                  f"{p*100:>5.1f}% [{lo*100:>5.1f}%, {hi*100:>5.1f}%]")
        print()

    # Per-model aggregated by (variant, condition)
    print("\n" + "=" * 90)
    print("REBOUND RATE PER (MODEL, VARIANT, CONDITION)")
    print("=" * 90)
    models_order = sorted({k[0] for k in counts})
    print(f"{'Model':<22} {'Variant':<32} {'cond':<14} {'k/n':>10} {'rate':>7}")
    for m in models_order:
        for v in variants_order:
            for c in conds_order:
                d = counts.get((m, v, c))
                if not d:
                    continue
                p, lo, hi = wilson_ci(d["correct"], d["total"])
                print(f"  {m:<20} {v:<32} {c:<14} {d['correct']:>3}/{d['total']:>3}  "
                      f"{p*100:>5.1f}%")
        print()

    # Paired McNemar test: same case, different conditions
    # Pair canonical_T2 vs null, and own_T2 vs null
    print("\n" + "=" * 90)
    print("PAIRED MCNEMAR TESTS")
    print("=" * 90)
    case_grades = defaultdict(dict)  # (model, variant, index) -> {cond: grade}
    for r in rows:
        case_grades[(r["model"], r["variant"], r["index"])][r["condition"]] = r.get("grade")

    print("\ncanonical_T2 vs null:")
    print(f"  {'cell':<46} {'b (can-only)':>12} {'c (null-only)':>13} "
          f"{'both-CORR':>10} {'both-INC':>10} {'McNemar p':>11}")
    for m in models_order:
        for v in variants_order:
            b = c = both_corr = both_inc = 0
            for k, grds in case_grades.items():
                if k[0] != m or k[1] != v: continue
                ca = grds.get("canonical_T2"); nu = grds.get("null")
                if ca is None or nu is None: continue
                if ca == "CORRECT" and nu == "INCORRECT": b += 1
                elif ca == "INCORRECT" and nu == "CORRECT": c += 1
                elif ca == "CORRECT" and nu == "CORRECT": both_corr += 1
                elif ca == "INCORRECT" and nu == "INCORRECT": both_inc += 1
            p = mcnemar_p(b, c)
            print(f"  {m+'/'+v:<46} {b:>12} {c:>13} {both_corr:>10} {both_inc:>10} {p:>11.3f}")

    print("\nown_T2 vs null:")
    print(f"  {'cell':<46} {'b (own-only)':>12} {'c (null-only)':>13} "
          f"{'both-CORR':>10} {'both-INC':>10} {'McNemar p':>11}")
    for m in models_order:
        for v in [vv for vv in variants_order if vv != "kernel_variant"]:
            b = c = both_corr = both_inc = 0
            for k, grds in case_grades.items():
                if k[0] != m or k[1] != v: continue
                ow = grds.get("own_T2"); nu = grds.get("null")
                if ow is None or nu is None: continue
                if ow == "CORRECT" and nu == "INCORRECT": b += 1
                elif ow == "INCORRECT" and nu == "CORRECT": c += 1
                elif ow == "CORRECT" and nu == "CORRECT": both_corr += 1
                elif ow == "INCORRECT" and nu == "INCORRECT": both_inc += 1
            p = mcnemar_p(b, c)
            print(f"  {m+'/'+v:<46} {b:>12} {c:>13} {both_corr:>10} {both_inc:>10} {p:>11.3f}")

    print("\nown_T2 vs canonical_T2:")
    print(f"  {'cell':<46} {'b (own-only)':>12} {'c (can-only)':>13} "
          f"{'both-CORR':>10} {'both-INC':>10} {'McNemar p':>11}")
    for m in models_order:
        for v in [vv for vv in variants_order if vv != "kernel_variant"]:
            b = c = both_corr = both_inc = 0
            for k, grds in case_grades.items():
                if k[0] != m or k[1] != v: continue
                ow = grds.get("own_T2"); ca = grds.get("canonical_T2")
                if ow is None or ca is None: continue
                if ow == "CORRECT" and ca == "INCORRECT": b += 1
                elif ow == "INCORRECT" and ca == "CORRECT": c += 1
                elif ow == "CORRECT" and ca == "CORRECT": both_corr += 1
                elif ow == "INCORRECT" and ca == "INCORRECT": both_inc += 1
            p = mcnemar_p(b, c)
            print(f"  {m+'/'+v:<46} {b:>12} {c:>13} {both_corr:>10} {both_inc:>10} {p:>11.3f}")


if __name__ == "__main__":
    main()
