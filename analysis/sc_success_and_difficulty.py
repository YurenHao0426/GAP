"""Two follow-up analyses (zero API):
1. Per-model self-correction success rate: P(correct | SC) vs P(correct | no SC)
2. Difficulty-stratified surface vs kernel dichotomy
"""
from __future__ import annotations
import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from structural_overlap import find_variant_file, load_problems, RESULTS_DIR, SURFACE_VARIANTS
from self_correction import has_self_correction


# ----------------- 1. SC success rate per model -----------------

def sc_success_rate():
    base = RESULTS_DIR
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])

    print("=" * 80)
    print("PER-MODEL SELF-CORRECTION SUCCESS RATE")
    print("(does an SC attempt improve probability of being correct?)")
    print("=" * 80)
    print()

    rows = []
    for m in models:
        mdir = base / m
        # Aggregate over all variants
        n_sc_correct = 0
        n_sc_total = 0
        n_nosc_correct = 0
        n_nosc_total = 0
        for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
            vp = find_variant_file(mdir, v)
            if not vp: continue
            for p in load_problems(vp):
                text = (p.get("solve") or {}).get("solution") or ""
                if not text: continue
                correct = p.get("correct")
                if correct is None: continue
                if has_self_correction(text):
                    n_sc_total += 1
                    if correct: n_sc_correct += 1
                else:
                    n_nosc_total += 1
                    if correct: n_nosc_correct += 1
        if n_sc_total < 5 or n_nosc_total < 5:
            continue
        p_sc = n_sc_correct / n_sc_total
        p_nosc = n_nosc_correct / n_nosc_total
        delta = p_sc - p_nosc
        # Wilson 95% CI on each rate
        rows.append({
            "model": m,
            "sc_n": n_sc_total, "sc_correct": n_sc_correct, "p_sc": p_sc,
            "nosc_n": n_nosc_total, "nosc_correct": n_nosc_correct, "p_nosc": p_nosc,
            "delta": delta,
        })

    rows.sort(key=lambda r: -r["sc_n"])
    print(f"{'Model':<22} {'#SC trials':>11} {'P(corr|SC)':>12} {'P(corr|noSC)':>13} {'Δ':>9}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model']:<22} {r['sc_n']:>11} "
              f"{r['p_sc']*100:>10.1f}% {r['p_nosc']*100:>11.1f}% "
              f"{r['delta']*100:>+7.1f}pp")

    json.dump(rows, open(THIS_DIR / "sc_success_per_model.json", "w"), indent=2)
    return rows


# ----------------- 2. Difficulty stratified dichotomy -----------------

DATASET_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")

def load_difficulty_metadata():
    """Per-problem difficulty assignment using year/section/index heuristic.

    Per the paper's existing exposition, we derive Easy/Medium/Hard from the
    problem index (1-2 = Easy, 3-4 = Medium, 5-6 = Hard, 7-8 = extra-hard tail)
    because the dataset's `difficulty` field is heterogeneous.
    """
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        if not idx: continue
        # Extract problem number from "YEAR-PART-NUM"
        parts = idx.split("-")
        if len(parts) != 3: continue
        try:
            num = int(parts[2])
        except ValueError:
            continue
        if num <= 2: bucket = "Easy"
        elif num <= 4: bucket = "Medium"
        elif num <= 6: bucket = "Hard"
        else: bucket = "ExtraHard"
        out[idx] = bucket
    return out


def difficulty_stratified_dichotomy():
    print("\n\n" + "=" * 80)
    print("DIFFICULTY-STRATIFIED ACCURACY (mean across 18 models)")
    print("Easy/Medium/Hard buckets defined by problem index 1-2/3-4/5-6")
    print("=" * 80)
    print()

    diff = load_difficulty_metadata()
    base = RESULTS_DIR
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])

    # buckets[(model, variant, difficulty)] = (n, n_correct)
    cells = defaultdict(lambda: [0, 0])
    for m in models:
        mdir = base / m
        for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
            vp = find_variant_file(mdir, v)
            if not vp: continue
            for p in load_problems(vp):
                idx = p.get("index")
                correct = p.get("correct")
                if idx is None or correct is None: continue
                bucket = diff.get(idx, "Unknown")
                cells[(m, v, bucket)][0] += 1
                if correct: cells[(m, v, bucket)][1] += 1

    # Aggregate per (variant, difficulty) by averaging per-model rates
    print(f"{'Variant':<24} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'XHard':>8}")
    print("-" * 60)
    for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
        row = {}
        for bucket in ["Easy", "Medium", "Hard", "ExtraHard"]:
            rates = []
            for m in models:
                n, c = cells.get((m, v, bucket), [0, 0])
                if n >= 5:
                    rates.append(c / n)
            row[bucket] = statistics.fmean(rates) * 100 if rates else None
        print(f"{v:<24} "
              f"{row['Easy']:>7.1f}% " if row['Easy'] is not None else f"{v:<24} {'-':>8}",
              end="")
        for bucket in ["Medium", "Hard", "ExtraHard"]:
            print(f"{row[bucket]:>7.1f}% " if row[bucket] is not None else f"{'-':>8}", end="")
        print()

    # Compute Δ_orig→KV per difficulty bucket
    print(f"\n--- Δ original → KV per difficulty bucket ---")
    for bucket in ["Easy", "Medium", "Hard", "ExtraHard"]:
        orig_rates = []
        kv_rates = []
        for m in models:
            no, co = cells.get((m, "original", bucket), [0, 0])
            nk, ck = cells.get((m, "kernel_variant", bucket), [0, 0])
            if no >= 5 and nk >= 5:
                orig_rates.append(co / no)
                kv_rates.append(ck / nk)
        if orig_rates:
            mo = statistics.fmean(orig_rates) * 100
            mk = statistics.fmean(kv_rates) * 100
            print(f"  {bucket:<10} orig={mo:5.1f}%  kv={mk:5.1f}%  Δ={mk-mo:+.1f}pp")

    # Compute Δ_orig→GS per difficulty bucket
    print(f"\n--- Δ original → GS (surface, hardest renamer) per difficulty bucket ---")
    for bucket in ["Easy", "Medium", "Hard", "ExtraHard"]:
        orig_rates = []
        gs_rates = []
        for m in models:
            no, co = cells.get((m, "original", bucket), [0, 0])
            ng, cg = cells.get((m, "garbled_string", bucket), [0, 0])
            if no >= 5 and ng >= 5:
                orig_rates.append(co / no)
                gs_rates.append(cg / ng)
        if orig_rates:
            mo = statistics.fmean(orig_rates) * 100
            mg = statistics.fmean(gs_rates) * 100
            print(f"  {bucket:<10} orig={mo:5.1f}%  GS={mg:5.1f}%  Δ={mg-mo:+.1f}pp")


def main():
    sc_success_rate()
    difficulty_stratified_dichotomy()


if __name__ == "__main__":
    main()
