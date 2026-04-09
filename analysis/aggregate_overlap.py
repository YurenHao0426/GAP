"""Aggregate structural_overlap results by variant type and by model.

Produces a clean rebuttal table.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from collections import defaultdict

RESULTS = Path("/home/yurenh2/gap/analysis/structural_overlap_results.json")
SHORT = {"descriptive_long":"DL","descriptive_long_confusing":"DLC",
         "descriptive_long_misleading":"DLM","garbled_string":"GS"}


def main():
    cells = json.load(open(RESULTS))
    print(f"Loaded {len(cells)} cells.\n")

    # Per-variant aggregate
    per_variant = defaultdict(list)
    for c in cells:
        per_variant[c["variant"]].append(c)

    print("=" * 90)
    print("HEADLINE TABLE: Surface variants — stable vs brittle structural overlap")
    print("(token Jaccard on canonicalized trajectories, drift cases only)")
    print("=" * 90)
    print(f"\n{'Variant':<6} {'#cells':>7} {'#dir+':>6} {'#p<.05':>8} "
          f"{'med-d':>7} {'mean-d':>7} {'mean-dlt':>9} "
          f"{'mean-stbl':>10} {'mean-brit':>10} {'mean-noise':>11} "
          f"{'mean-collapse%':>14}")
    print("-" * 100)
    for v, cs in per_variant.items():
        ds = [c["metrics"]["token_jaccard"]["cohens_d"] for c in cs]
        ps = [c["metrics"]["token_jaccard"]["p_two_sided"] for c in cs]
        n_pos = sum(1 for d in ds if d > 0)
        n_sig = sum(1 for p in ps if p < 0.05)
        deltas = [c["metrics"]["token_jaccard"]["delta_median"] for c in cs]
        stbl = [c["metrics"]["token_jaccard"]["stable_median"] for c in cs]
        brit = [c["metrics"]["token_jaccard"]["brittle_median"] for c in cs]
        noise = [c["metrics"]["token_jaccard"]["noise_floor_median"] for c in cs
                 if c["metrics"]["token_jaccard"].get("noise_floor_median") is not None]
        collapse = [c["brittle_collapse_rate"] for c in cs]
        print(f"{SHORT[v]:<6} {len(cs):>7} {n_pos:>6} {n_sig:>8} "
              f"{statistics.median(ds):>+7.2f} {statistics.fmean(ds):>+7.2f} "
              f"{statistics.fmean(deltas):>+9.4f} "
              f"{statistics.fmean(stbl):>10.3f} {statistics.fmean(brit):>10.3f} "
              f"{statistics.fmean(noise):>11.3f} "
              f"{statistics.fmean(collapse)*100:>13.1f}%")

    # Variant-aggregate (across all models, n-weighted)
    print("\n" + "=" * 90)
    print("ALL CELLS (18 models × 4 surface variants)")
    print("=" * 90)
    all_d = [c["metrics"]["token_jaccard"]["cohens_d"] for c in cells]
    all_p = [c["metrics"]["token_jaccard"]["p_two_sided"] for c in cells]
    print(f"  cells:                  {len(cells)}")
    print(f"  direction-positive:     {sum(1 for d in all_d if d>0)}/{len(cells)}")
    print(f"  p<0.05:                 {sum(1 for p in all_p if p<0.05)}/{len(cells)}")
    print(f"  p<0.001:                {sum(1 for p in all_p if p<0.001)}/{len(cells)}")
    print(f"  p<1e-6:                 {sum(1 for p in all_p if p<1e-6)}/{len(cells)}")
    print(f"  Cohen's d median:       {statistics.median(all_d):+.3f}")
    print(f"  Cohen's d mean:         {statistics.fmean(all_d):+.3f}")
    print(f"  Cohen's d range:        [{min(all_d):+.2f}, {max(all_d):+.2f}]")

    # Per-model aggregate (averaged across 4 surface variants)
    per_model = defaultdict(list)
    for c in cells:
        per_model[c["model"]].append(c)
    print("\n" + "=" * 90)
    print("PER MODEL (averaged across 4 surface variants)")
    print("=" * 90)
    print(f"\n{'Model':<25} {'mean-d':>7} {'mean-stbl':>10} {'mean-brit':>10} "
          f"{'mean-coll%':>11} {'min-p':>9}")
    print("-" * 80)
    rows = []
    for m, cs in per_model.items():
        if len(cs) == 0: continue
        d = statistics.fmean(c["metrics"]["token_jaccard"]["cohens_d"] for c in cs)
        s = statistics.fmean(c["metrics"]["token_jaccard"]["stable_median"] for c in cs)
        b = statistics.fmean(c["metrics"]["token_jaccard"]["brittle_median"] for c in cs)
        col = statistics.fmean(c["brittle_collapse_rate"] for c in cs) * 100
        mp = min(c["metrics"]["token_jaccard"]["p_two_sided"] for c in cs)
        rows.append((m, d, s, b, col, mp))
    for r in sorted(rows, key=lambda r: -r[1]):
        print(f"{r[0]:<25} {r[1]:>+7.2f} {r[2]:>10.3f} {r[3]:>10.3f} {r[4]:>10.1f}% {r[5]:>9.1e}")


if __name__ == "__main__":
    main()
