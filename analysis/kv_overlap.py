"""Kernel-variant structural-overlap analysis (label-free).

Unlike surface variants, kernel variants change the math, so we cannot use the
model's own original-correct trajectory as a reference. Instead we use the
dataset's canonical kernel-variant solution as the reference.

Hypothesis: stable (correct on KV) trajectories have higher structural overlap
with the canonical KV solution than brittle (wrong on KV) trajectories.

For comparability we also recompute the surface analyses using the same
'overlap with canonical solution' metric, so we can compare apples-to-apples
the magnitude of stable-vs-brittle gap between surface and kernel.
"""
from __future__ import annotations
import json
import os
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Reuse helpers from the sibling module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from structural_overlap import (
    DATASET_DIR, RESULTS_DIR,
    load_problems, find_variant_file,
    canonicalize_text, normalize_whitespace,
    tokens, bigrams, jaccard, extract_math_blocks,
    metric_token_jaccard, metric_bigram_jaccard,
    metric_directional_coverage, metric_equation_jaccard,
    mann_whitney_u, bootstrap_ci_cohens_d,
    is_collapse, COLLAPSE_MIN_CHARS, COLLAPSE_RATIO,
    SURFACE_VARIANTS,
)


def load_dataset_variant_solutions() -> dict:
    """Returns: {problem_index: {variant_name: canonical_solution_text}}.

    Includes 'original' (from top-level field) plus all 5 variants.
    """
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        cell = {"original": d.get("solution") or "",
                "_problem_type": d.get("problem_type")}
        for v, vd in d.get("variants", {}).items():
            if isinstance(vd, dict):
                cell[v] = vd.get("solution") or ""
        out[idx] = cell
    return out


def load_dataset_maps() -> dict:
    """Mirrors structural_overlap.load_dataset_maps but localized for safety."""
    out = {}
    for f in sorted(DATASET_DIR.glob("*.json")):
        d = json.load(open(f))
        idx = d.get("index")
        variants = d.get("variants", {})
        cell = {}
        for v in SURFACE_VARIANTS:
            vd = variants.get(v, {})
            mp_str = vd.get("map")
            if isinstance(mp_str, str):
                try:
                    mp = eval(mp_str, {"__builtins__": {}}, {})
                    if isinstance(mp, dict):
                        cell[v] = {str(k): str(v) for k, v in mp.items()}
                except Exception:
                    pass
            elif isinstance(mp_str, dict):
                cell[v] = {str(k): str(v) for k, v in mp_str.items()}
        out[idx] = cell
    return out


# ---------- Cell analyzer ----------

def analyze_kv_cell(model_name: str, model_dir: Path,
                    canonical_solutions: dict) -> Optional[dict]:
    """Compare model's KV trajectory to dataset canonical KV solution.

    No canonicalization (no rename map for KV — variables match by construction).
    """
    orig_path = find_variant_file(model_dir, "original")
    var_path = find_variant_file(model_dir, "kernel_variant")
    if not orig_path or not var_path:
        return None
    orig_by = {p["index"]: p for p in load_problems(orig_path)}
    var_by = {p["index"]: p for p in load_problems(var_path)}

    pairs_stable_drift = []
    pairs_brittle_drift = []
    n_brittle_collapse = 0
    n_stable_collapse = 0

    for idx in set(orig_by) & set(var_by):
        po, pv = orig_by[idx], var_by[idx]
        if po.get("correct") is not True:
            continue  # Restrict to "model already gets the original"
        var_correct = pv.get("correct")
        if var_correct is None:
            continue
        var_text = (pv.get("solve") or {}).get("solution") or ""
        if not var_text:
            continue
        canon_kv = canonical_solutions.get(idx, {}).get("kernel_variant", "")
        if not canon_kv or len(canon_kv) < 200:
            continue
        # Collapse rule: variant text < 200 chars OR < 25% of canonical solution
        collapse = (len(var_text) < COLLAPSE_MIN_CHARS or
                    len(var_text) < COLLAPSE_RATIO * len(canon_kv))
        sample = {"index": idx, "var_text": var_text, "canon": canon_kv}
        if var_correct is True:
            if collapse:
                n_stable_collapse += 1
            else:
                pairs_stable_drift.append(sample)
        else:
            if collapse:
                n_brittle_collapse += 1
            else:
                pairs_brittle_drift.append(sample)

    if not pairs_stable_drift or not pairs_brittle_drift:
        return None

    metrics = {
        "token_jaccard": metric_token_jaccard,
        "bigram_jaccard": metric_bigram_jaccard,
        "equation_jaccard": metric_equation_jaccard,
        "directional_coverage": metric_directional_coverage,
    }

    out = {
        "model": model_name,
        "variant": "kernel_variant",
        "n_stable_drift": len(pairs_stable_drift),
        "n_brittle_drift": len(pairs_brittle_drift),
        "n_brittle_collapse": n_brittle_collapse,
        "n_stable_collapse": n_stable_collapse,
        "brittle_collapse_rate": n_brittle_collapse /
                                 max(1, n_brittle_collapse + len(pairs_brittle_drift)),
        "metrics": {},
    }
    for mname, mfn in metrics.items():
        s_vals = [mfn(p["var_text"], p["canon"]) for p in pairs_stable_drift]
        b_vals = [mfn(p["var_text"], p["canon"]) for p in pairs_brittle_drift]
        U, p = mann_whitney_u(s_vals, b_vals)
        sm, bm = statistics.fmean(s_vals), statistics.fmean(b_vals)
        ssd = statistics.pstdev(s_vals) if len(s_vals) > 1 else 0
        bsd = statistics.pstdev(b_vals) if len(b_vals) > 1 else 0
        pooled = (((len(s_vals)-1)*ssd**2 + (len(b_vals)-1)*bsd**2)
                  / max(1, len(s_vals)+len(b_vals)-2)) ** 0.5
        d = (sm - bm) / pooled if pooled > 0 else 0.0
        out["metrics"][mname] = {
            "stable_median": statistics.median(s_vals),
            "stable_mean": sm,
            "brittle_median": statistics.median(b_vals),
            "brittle_mean": bm,
            "delta_median": statistics.median(s_vals) - statistics.median(b_vals),
            "cohens_d": d,
            "U": U,
            "p_two_sided": p,
        }
    # Headline bootstrap
    s_vals = [metric_token_jaccard(p["var_text"], p["canon"]) for p in pairs_stable_drift]
    b_vals = [metric_token_jaccard(p["var_text"], p["canon"]) for p in pairs_brittle_drift]
    d_lo, d_hi = bootstrap_ci_cohens_d(s_vals, b_vals, n_iter=400)
    out["metrics"]["token_jaccard"]["cohens_d_ci"] = [d_lo, d_hi]
    return out


# ---------- Surface re-analysis with canonical reference ----------

def analyze_surface_cell_against_canonical(model_name: str, variant: str,
                                           model_dir: Path,
                                           canonical_solutions: dict) -> Optional[dict]:
    """Compare model variant trajectory to dataset canonical variant solution.

    For comparability with KV. No rename canonicalization needed since both
    sides use the same variant naming.
    """
    var_path = find_variant_file(model_dir, variant)
    orig_path = find_variant_file(model_dir, "original")
    if not var_path or not orig_path:
        return None
    var_by = {p["index"]: p for p in load_problems(var_path)}
    orig_by = {p["index"]: p for p in load_problems(orig_path)}

    pairs_stable, pairs_brittle = [], []
    n_brittle_collapse = 0
    for idx in set(var_by):
        if idx not in orig_by:
            continue
        if orig_by[idx].get("correct") is not True:
            continue  # restrict to model-knows-original
        pv = var_by[idx]
        var_correct = pv.get("correct")
        if var_correct is None:
            continue
        var_text = (pv.get("solve") or {}).get("solution") or ""
        if not var_text:
            continue
        canon_var = canonical_solutions.get(idx, {}).get(variant, "")
        if not canon_var or len(canon_var) < 200:
            continue
        if (len(var_text) < COLLAPSE_MIN_CHARS or
                len(var_text) < COLLAPSE_RATIO * len(canon_var)):
            if var_correct is False:
                n_brittle_collapse += 1
            continue
        sample = {"index": idx, "var_text": var_text, "canon": canon_var}
        if var_correct is True:
            pairs_stable.append(sample)
        else:
            pairs_brittle.append(sample)

    if not pairs_stable or not pairs_brittle:
        return None

    metrics = {
        "token_jaccard": metric_token_jaccard,
        "bigram_jaccard": metric_bigram_jaccard,
        "equation_jaccard": metric_equation_jaccard,
        "directional_coverage": metric_directional_coverage,
    }
    out = {
        "model": model_name,
        "variant": variant,
        "n_stable_drift": len(pairs_stable),
        "n_brittle_drift": len(pairs_brittle),
        "n_brittle_collapse": n_brittle_collapse,
        "brittle_collapse_rate": n_brittle_collapse /
                                 max(1, n_brittle_collapse + len(pairs_brittle)),
        "metrics": {},
    }
    for mname, mfn in metrics.items():
        s_vals = [mfn(p["var_text"], p["canon"]) for p in pairs_stable]
        b_vals = [mfn(p["var_text"], p["canon"]) for p in pairs_brittle]
        U, p = mann_whitney_u(s_vals, b_vals)
        sm, bm = statistics.fmean(s_vals), statistics.fmean(b_vals)
        ssd = statistics.pstdev(s_vals) if len(s_vals) > 1 else 0
        bsd = statistics.pstdev(b_vals) if len(b_vals) > 1 else 0
        pooled = (((len(s_vals)-1)*ssd**2 + (len(b_vals)-1)*bsd**2)
                  / max(1, len(s_vals)+len(b_vals)-2)) ** 0.5
        d = (sm - bm) / pooled if pooled > 0 else 0.0
        out["metrics"][mname] = {
            "stable_median": statistics.median(s_vals),
            "stable_mean": sm,
            "brittle_median": statistics.median(b_vals),
            "brittle_mean": bm,
            "delta_median": statistics.median(s_vals) - statistics.median(b_vals),
            "cohens_d": d,
            "U": U,
            "p_two_sided": p,
        }
    return out


def main():
    print("Loading canonical solutions ...")
    canon = load_dataset_variant_solutions()
    print(f"  loaded {len(canon)} problems")

    all_models = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])

    kv_results = []
    surface_results = []

    print(f"\n{'KERNEL VARIANT — variant trajectory vs canonical KV solution':<70}")
    print(f"{'Cell':<32} {'nSd':>4} {'nBd':>4} {'col%':>5} "
          f"{'sMed':>6} {'bMed':>6} {'d':>6} {'p':>9}")
    print("-" * 90)
    for m in all_models:
        mdir = RESULTS_DIR / m
        if not mdir.exists():
            continue
        res = analyze_kv_cell(m, mdir, canon)
        if res is None:
            continue
        kv_results.append(res)
        md = res["metrics"]["token_jaccard"]
        print(f"{m+' / KV':<32} {res['n_stable_drift']:>4} {res['n_brittle_drift']:>4} "
              f"{res['brittle_collapse_rate']*100:>4.0f}% "
              f"{md['stable_median']:>6.3f} {md['brittle_median']:>6.3f} "
              f"{md['cohens_d']:>+6.2f} {md['p_two_sided']:>9.1e}")

    print(f"\n{'SURFACE VARIANT — variant trajectory vs canonical variant solution':<70}")
    print(f"{'Cell':<46} {'nSd':>4} {'nBd':>4} {'col%':>5} "
          f"{'sMed':>6} {'bMed':>6} {'d':>6} {'p':>9}")
    print("-" * 95)
    for m in all_models:
        mdir = RESULTS_DIR / m
        if not mdir.exists():
            continue
        for v in SURFACE_VARIANTS:
            res = analyze_surface_cell_against_canonical(m, v, mdir, canon)
            if res is None:
                continue
            surface_results.append(res)
            md = res["metrics"]["token_jaccard"]
            print(f"{m+' / '+v:<46} {res['n_stable_drift']:>4} {res['n_brittle_drift']:>4} "
                  f"{res['brittle_collapse_rate']*100:>4.0f}% "
                  f"{md['stable_median']:>6.3f} {md['brittle_median']:>6.3f} "
                  f"{md['cohens_d']:>+6.2f} {md['p_two_sided']:>9.1e}")

    # Save
    json.dump(kv_results, open("/home/yurenh2/gap/analysis/kv_overlap_results.json", "w"), indent=2)
    json.dump(surface_results, open("/home/yurenh2/gap/analysis/surface_canonical_results.json", "w"), indent=2)

    # Aggregate compare
    print("\n" + "=" * 80)
    print("AGGREGATE: surface (vs canonical) vs kernel (vs canonical)")
    print("=" * 80)
    for tag, results in [("surface", surface_results), ("kernel", kv_results)]:
        ds = [c["metrics"]["token_jaccard"]["cohens_d"] for c in results]
        ps = [c["metrics"]["token_jaccard"]["p_two_sided"] for c in results]
        col = [c["brittle_collapse_rate"] for c in results]
        if not ds:
            continue
        print(f"{tag:<8}  cells={len(ds):>3}  d_pos={sum(1 for d in ds if d>0):>3}/{len(ds):<3}  "
              f"p<.05={sum(1 for p in ps if p<0.05):>3}/{len(ps):<3}  "
              f"d_med={statistics.median(ds):+.2f}  d_mean={statistics.fmean(ds):+.2f}  "
              f"collapse_mean={statistics.fmean(col)*100:.1f}%")


if __name__ == "__main__":
    main()
