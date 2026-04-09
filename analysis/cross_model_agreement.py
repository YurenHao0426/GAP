"""Cross-model agreement analysis: which problems are universally hard?

For each (variant, problem) cell, count how many models fail (correct=False).
Identify "universally hard" problems (failed by ≥80% of models on the variant)
and "universally easy" (correct by ≥80% on the variant). Then check whether
the universally hard *flip set* is dominated by certain topics, problem types,
or years.

Outputs:
- Per-variant histogram of failure counts
- "Universal flip" cases: original correct by ≥80% of models, variant wrong by ≥80%
- These are the cleanest signals of variant-induced fragility because they
  rule out problem-specific quirks
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from structural_overlap import find_variant_file, load_problems, RESULTS_DIR, SURFACE_VARIANTS

DATASET_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")


def load_metadata():
    """Load problem-level metadata: type, tag, difficulty, year."""
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        out[idx] = {
            "type": d.get("type"),
            "tag": d.get("tag"),
            "difficulty": d.get("difficulty"),
            "problem_type": d.get("problem_type"),
            "year": int(idx.split("-")[0]) if idx and "-" in idx else None,
        }
    return out


def main():
    base = RESULTS_DIR
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])
    print(f"Loading {len(models)} models ...")
    metadata = load_metadata()

    # correct_table[(variant, idx)][model] = bool
    correct_table = defaultdict(dict)
    for m in models:
        mdir = base / m
        for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
            vp = find_variant_file(mdir, v)
            if not vp:
                continue
            for p in load_problems(vp):
                idx = p.get("index")
                correct = p.get("correct")
                if idx is None or correct is None:
                    continue
                correct_table[(v, idx)][m] = correct

    print(f"Loaded {len(correct_table)} (variant, problem) cells.\n")

    # Per-variant histogram of correct counts (out of N models)
    print("=== HISTOGRAM OF CORRECT-COUNT ACROSS MODELS ===")
    print("(How many models get each problem right per variant)\n")
    print(f"{'Variant':<24} {'mean correct/N':>16} {'median':>9} {'#unanimous-fail':>17} {'#unanimous-pass':>17}")
    print("-" * 90)
    for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
        cells = [d for (vv, idx), d in correct_table.items() if vv == v]
        if not cells:
            continue
        counts = [sum(1 for vv in cell.values() if vv) / len(cell) for cell in cells]
        unanimous_fail = sum(1 for cell in cells if not any(cell.values()) and len(cell) >= 3)
        unanimous_pass = sum(1 for cell in cells if all(cell.values()) and len(cell) >= 3)
        import statistics
        print(f"{v:<24} {statistics.fmean(counts)*100:>14.1f}%  {statistics.median(counts)*100:>7.1f}% "
              f"{unanimous_fail:>17} {unanimous_pass:>17}")

    # Universal flip cases: original correct by ≥80% of models, variant wrong by ≥80%
    print(f"\n\n=== UNIVERSAL FLIP CASES (orig ≥80% correct, variant ≥80% wrong) ===\n")
    print("These are the cleanest signals of variant-induced fragility.\n")
    print(f"{'Variant':<24} {'# universal flips':>20}")
    print("-" * 50)
    universal_flips = defaultdict(list)
    for v in SURFACE_VARIANTS + ["kernel_variant"]:
        for idx in {ii for (vv, ii) in correct_table if vv == "original"}:
            orig_cell = correct_table.get(("original", idx), {})
            var_cell = correct_table.get((v, idx), {})
            common = set(orig_cell) & set(var_cell)
            if len(common) < 5: continue
            orig_rate = sum(1 for m in common if orig_cell[m]) / len(common)
            var_rate = sum(1 for m in common if var_cell[m]) / len(common)
            if orig_rate >= 0.80 and var_rate <= 0.20:
                universal_flips[v].append((idx, orig_rate, var_rate))
        print(f"{v:<24} {len(universal_flips[v]):>20}")

    # Topic / problem_type / difficulty / year breakdown for universal flips
    print(f"\n\n=== TOPIC BREAKDOWN OF UNIVERSAL FLIPS ===\n")
    for v in SURFACE_VARIANTS + ["kernel_variant"]:
        if not universal_flips[v]: continue
        print(f"--- {v} ({len(universal_flips[v])} universal flips) ---")
        topics = Counter()
        ptypes = Counter()
        difficulties = Counter()
        years = Counter()
        for idx, _, _ in universal_flips[v]:
            md = metadata.get(idx, {})
            tag = md.get("tag")
            # tag may be a list (multi-tag) or a string
            if isinstance(tag, list):
                for t in tag: topics[t] += 1
            elif tag:
                topics[tag] += 1
            else:
                topics["?"] += 1
            ptypes[md.get("problem_type") or "?"] += 1
            diff = md.get("difficulty")
            if isinstance(diff, list): diff = diff[0] if diff else "?"
            difficulties[diff or "?"] += 1
            year = md.get("year")
            if year:
                # Bin years by decade
                decade = (year // 10) * 10
                years[f"{decade}s"] += 1
        print(f"  topics:      {dict(topics.most_common(8))}")
        print(f"  problem_type:{dict(ptypes)}")
        print(f"  difficulties:{dict(difficulties.most_common(6))}")
        print(f"  decades:     {dict(sorted(years.items()))}")
        print()

    # Save universal flips for later analysis
    out = {v: [{"index": idx, "orig_rate": o, "var_rate": vr}
               for (idx, o, vr) in flips]
           for v, flips in universal_flips.items()}
    json.dump(out, open(THIS_DIR / "universal_flips.json", "w"), indent=2)
    print(f"\nSaved -> analysis/universal_flips.json")

    # Topic-stratified analysis: failure rate per topic per variant
    print(f"\n\n=== ACCURACY BY TOPIC × VARIANT (mean across models) ===\n")
    by_topic_variant = defaultdict(lambda: defaultdict(list))
    for (v, idx), cell in correct_table.items():
        md = metadata.get(idx, {})
        tag = md.get("tag")
        if not tag or len(cell) < 5: continue
        # If multiple tags, attribute the same rate to each — keeps it simple
        topics_for_problem = tag if isinstance(tag, list) else [tag]
        rate = sum(1 for vv in cell.values() if vv) / len(cell)
        for t in topics_for_problem:
            by_topic_variant[t][v].append(rate)

    topics_to_show = ["ALG", "ANA", "NT", "COMB", "GEO"]
    print(f"{'Topic':<8}", end="")
    for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
        short = {"original":"orig","descriptive_long":"DL","descriptive_long_confusing":"DLC",
                 "descriptive_long_misleading":"DLM","garbled_string":"GS","kernel_variant":"KV"}[v]
        print(f"  {short:>5}", end="")
    print("  Δ_orig→KV")
    print("-" * 70)
    for t in topics_to_show:
        if t not in by_topic_variant: continue
        row = by_topic_variant[t]
        if "original" not in row: continue
        orig_mean = statistics.fmean(row["original"]) * 100
        print(f"{t:<8}", end="")
        for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
            if v in row:
                m = statistics.fmean(row[v]) * 100
                print(f"  {m:>4.1f}%", end="")
            else:
                print(f"  {'-':>5}", end="")
        kv_mean = statistics.fmean(row.get("kernel_variant", [0])) * 100
        print(f"   {kv_mean - orig_mean:+5.1f}pp")


if __name__ == "__main__":
    main()
