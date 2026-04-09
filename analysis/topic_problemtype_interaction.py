"""KV fragility broken down by Topic × Problem-type (proof vs calculation)."""
from __future__ import annotations
import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from structural_overlap import find_variant_file, load_problems, RESULTS_DIR, SURFACE_VARIANTS

DATASET_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")


def load_metadata():
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        if not idx: continue
        out[idx] = {
            "tag": d.get("tag"),
            "problem_type": d.get("problem_type"),
        }
    return out


def main():
    metadata = load_metadata()
    base = RESULTS_DIR
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])

    # cells[(topic, ptype, model, variant)] = (n, n_correct)
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
                md = metadata.get(idx, {})
                tag = md.get("tag")
                ptype = md.get("problem_type")
                if not tag or not ptype: continue
                tags = tag if isinstance(tag, list) else [tag]
                for t in tags:
                    if t not in ["ALG", "ANA", "NT", "COMB", "GEO"]: continue
                    cells[(t, ptype, m, v)][0] += 1
                    if correct: cells[(t, ptype, m, v)][1] += 1

    print("=" * 80)
    print("ACCURACY BY TOPIC × PROBLEM-TYPE × VARIANT (mean across 18 models)")
    print("=" * 80)
    print()

    for ptype in ["proof", "calculation"]:
        print(f"\n--- {ptype.upper()} ---\n")
        print(f"{'Topic':<6}", end="")
        for v in ["original", "garbled_string", "kernel_variant"]:
            short = {"original":"orig","garbled_string":"GS","kernel_variant":"KV"}[v]
            print(f"  {short:>6}", end="")
        print(f"  {'Δ_GS':>7} {'Δ_KV':>7}")
        print("-" * 50)
        for t in ["ALG", "ANA", "NT", "COMB", "GEO"]:
            orig_rates = []
            gs_rates = []
            kv_rates = []
            for m in models:
                no, co = cells.get((t, ptype, m, "original"), [0, 0])
                ng, cg = cells.get((t, ptype, m, "garbled_string"), [0, 0])
                nk, ck = cells.get((t, ptype, m, "kernel_variant"), [0, 0])
                if no >= 5 and ng >= 5 and nk >= 5:
                    orig_rates.append(co / no)
                    gs_rates.append(cg / ng)
                    kv_rates.append(ck / nk)
            if not orig_rates: continue
            mo = statistics.fmean(orig_rates) * 100
            mg = statistics.fmean(gs_rates) * 100
            mk = statistics.fmean(kv_rates) * 100
            print(f"{t:<6}  {mo:>5.1f}% {mg:>5.1f}% {mk:>5.1f}%  {mg-mo:>+5.1f}pp {mk-mo:>+5.1f}pp")

    print("\n\n=== KEY DIFFERENTIAL: Δ KV by Topic for proof vs calculation ===\n")
    print(f"{'Topic':<6}  {'proof Δ':>10} {'calc Δ':>10} {'(calc - proof)':>16}")
    print("-" * 50)
    for t in ["ALG", "ANA", "NT", "COMB", "GEO"]:
        deltas = {}
        for ptype in ["proof", "calculation"]:
            orig_rates = []
            kv_rates = []
            for m in models:
                no, co = cells.get((t, ptype, m, "original"), [0, 0])
                nk, ck = cells.get((t, ptype, m, "kernel_variant"), [0, 0])
                if no >= 5 and nk >= 5:
                    orig_rates.append(co / no)
                    kv_rates.append(ck / nk)
            if orig_rates:
                deltas[ptype] = (statistics.fmean(kv_rates) - statistics.fmean(orig_rates)) * 100
        if "proof" in deltas and "calculation" in deltas:
            diff = deltas["calculation"] - deltas["proof"]
            print(f"{t:<6}  {deltas['proof']:>+9.1f}pp {deltas['calculation']:>+9.1f}pp {diff:>+15.1f}pp")
        elif "proof" in deltas:
            print(f"{t:<6}  {deltas['proof']:>+9.1f}pp {'-':>10} {'-':>16}")
        elif "calculation" in deltas:
            print(f"{t:<6}  {'-':>10} {deltas['calculation']:>+9.1f}pp {'-':>16}")


if __name__ == "__main__":
    main()
