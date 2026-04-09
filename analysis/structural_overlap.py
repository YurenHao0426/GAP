"""Stable-vs-Brittle structural overlap analysis (label-free).

Pipeline:
1. For each (model, surface_variant) cell, load original and variant trajectories.
2. Pull the deterministic rename map from /home/yurenh2/gap/putnam-bench-anon/dataset/.
3. Canonicalize both trajectories: replace variant variables with placeholders
   (via inverse rename map). Original trajectory: replace canonical variables
   with the same placeholders. Both texts then live in a shared placeholder space.
4. Compute multiple non-LLM structural metrics on (orig_canonical, var_canonical):
   - Token Jaccard
   - Bigram Jaccard
   - Equation-set Jaccard (math-block extraction)
   - Prefix Jaccard (first 30% of each canonical text)
5. Stratify by group (stable vs brittle) within each (model, variant) cell.
6. Mann-Whitney U test on each metric for stable vs brittle.

Surface variants only (rename map available). Kernel handled separately.
"""

from __future__ import annotations
import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import statistics

DATASET_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")
RESULTS_DIR = Path("/home/yurenh2/gap/results_new")

SURFACE_VARIANTS = ["descriptive_long", "descriptive_long_confusing",
                    "descriptive_long_misleading", "garbled_string"]


# ---------- I/O helpers ----------

def load_problems(path: Path) -> List[dict]:
    d = json.load(open(path))
    return d.get("problems") or d.get("detailed_results") or []


def find_variant_file(model_dir: Path, variant: str) -> Optional[Path]:
    files = sorted(os.listdir(model_dir))
    cands = [f for f in files
             if f.endswith(f"_{variant}.json")
             and "regraded" not in f and "comparison" not in f
             and not f.endswith(f"_{variant}2.json")]
    if not cands and variant == "garbled_string":
        cands = [f for f in files if f.endswith("_gs.json")]
    return model_dir / cands[0] if cands else None


def load_dataset_maps() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Returns: {problem_index: {variant: {orig_var_name: variant_var_name}}}"""
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
                # The map is stored as a Python repr string; eval it safely
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


# ---------- Canonicalization ----------

def canonicalize_text(text: str, var_to_placeholder: Dict[str, str]) -> str:
    """Replace each variable name in text with its canonical placeholder.

    Sort by length desc to avoid prefix collisions (e.g., 'xs' before 'x').
    Use word-boundary regex for ASCII-identifier-like names; literal replace
    for non-identifier names (like garbled strings, which are also alpha).
    """
    if not text:
        return ""
    # Sort longest-first to avoid 'al' eating into 'almondtree'
    items = sorted(var_to_placeholder.items(), key=lambda kv: -len(kv[0]))
    out = text
    for var, ph in items:
        if not var:
            continue
        # Use word-boundary so we only replace whole tokens. Variables in this
        # dataset are all alphanumeric.
        pat = r"(?<![A-Za-z0-9_])" + re.escape(var) + r"(?![A-Za-z0-9_])"
        out = re.sub(pat, ph, out)
    return out


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------- Tokenization ----------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

def tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def bigrams(toks: List[str]) -> List[str]:
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]


# ---------- Math block extraction ----------

_MATH_BLOCKS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\$(.+?)\$", re.DOTALL),
    re.compile(r"\\begin\{(?:equation|align|gather)\*?\}(.+?)\\end\{(?:equation|align|gather)\*?\}", re.DOTALL),
]

def extract_math_blocks(text: str, min_len: int = 8) -> List[str]:
    found = []
    for pat in _MATH_BLOCKS:
        found.extend(pat.findall(text or ""))
    # Lightweight normalization: collapse whitespace, strip
    out = [normalize_whitespace(b) for b in found if b.strip()]
    # Filter trivial fragments like '$n$', '$0$', '$x$' that saturate Jaccard
    return [b for b in out if len(b) >= min_len]


# ---------- Metrics ----------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def metric_token_jaccard(a: str, b: str) -> float:
    return jaccard(set(tokens(a)), set(tokens(b)))


def metric_bigram_jaccard(a: str, b: str) -> float:
    return jaccard(set(bigrams(tokens(a))), set(bigrams(tokens(b))))


def metric_prefix_token_jaccard(a: str, b: str, frac: float = 0.3) -> float:
    """Jaccard over the first frac of tokens from each side."""
    ta, tb = tokens(a), tokens(b)
    na, nb = max(1, int(len(ta) * frac)), max(1, int(len(tb) * frac))
    return jaccard(set(ta[:na]), set(tb[:nb]))


def metric_prefix_bigram_jaccard(a: str, b: str, frac: float = 0.3) -> float:
    ta, tb = tokens(a), tokens(b)
    na, nb = max(1, int(len(ta) * frac)), max(1, int(len(tb) * frac))
    return jaccard(set(bigrams(ta[:na])), set(bigrams(tb[:nb])))


def metric_equation_jaccard(a: str, b: str) -> float:
    ea = set(extract_math_blocks(a))
    eb = set(extract_math_blocks(b))
    return jaccard(ea, eb)


def metric_lcp_tokens(a: str, b: str) -> int:
    """Length of the longest common prefix of canonicalized token streams.

    Directly tests Codex's thesis 'early loss of structural overlap with the
    model's own original reasoning under renaming'. Larger LCP -> the model
    started its variant trajectory the same way it started the original.
    """
    ta, tb = tokens(a), tokens(b)
    n = min(len(ta), len(tb))
    i = 0
    while i < n and ta[i] == tb[i]:
        i += 1
    return i


def metric_lcp_normalized(a: str, b: str) -> float:
    """LCP length normalized by the shorter trajectory length, in [0, 1]."""
    ta, tb = tokens(a), tokens(b)
    n = min(len(ta), len(tb))
    if n == 0:
        return 0.0
    return metric_lcp_tokens(a, b) / n


def metric_lcp_first1k(a: str, b: str) -> float:
    """LCP length capped to first-1000-token comparison, normalized to [0, 1]."""
    ta, tb = tokens(a), tokens(b)
    ta, tb = ta[:1000], tb[:1000]
    n = min(len(ta), len(tb))
    if n == 0:
        return 0.0
    i = 0
    while i < n and ta[i] == tb[i]:
        i += 1
    return i / n


def metric_directional_coverage(a: str, b: str) -> float:
    """|tokens_a ∩ tokens_b| / |tokens_a|. Length-asymmetric.

    Reads as: 'what fraction of the original's vocabulary survives in the variant?'
    More robust to length differences than symmetric Jaccard.
    """
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def metric_window_token_jaccard(a: str, b: str, window: int = 600) -> float:
    """Jaccard restricted to the first `window` tokens on each side."""
    ta = tokens(a)[:window]
    tb = tokens(b)[:window]
    return jaccard(set(ta), set(tb))


def metric_window_bigram_jaccard(a: str, b: str, window: int = 600) -> float:
    ta = tokens(a)[:window]
    tb = tokens(b)[:window]
    return jaccard(set(bigrams(ta)), set(bigrams(tb)))


# ---------- Stat helpers ----------

def bootstrap_ci_delta_median(xs: List[float], ys: List[float],
                              n_iter: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI on median(xs) - median(ys)."""
    import random
    rng = random.Random(seed)
    if not xs or not ys:
        return float("nan"), float("nan")
    ds = []
    for _ in range(n_iter):
        rs = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        rb = [ys[rng.randrange(len(ys))] for _ in range(len(ys))]
        ds.append(statistics.median(rs) - statistics.median(rb))
    ds.sort()
    lo = ds[int(0.025 * n_iter)]
    hi = ds[int(0.975 * n_iter)]
    return lo, hi


def bootstrap_ci_cohens_d(xs: List[float], ys: List[float],
                          n_iter: int = 1000, seed: int = 0) -> Tuple[float, float]:
    import random
    rng = random.Random(seed)
    if len(xs) < 2 or len(ys) < 2:
        return float("nan"), float("nan")
    ds = []
    for _ in range(n_iter):
        rs = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        rb = [ys[rng.randrange(len(ys))] for _ in range(len(ys))]
        sm, bm = statistics.fmean(rs), statistics.fmean(rb)
        ssd = statistics.pstdev(rs)
        bsd = statistics.pstdev(rb)
        pooled = (((len(rs)-1)*ssd**2 + (len(rb)-1)*bsd**2)
                  / max(1, len(rs)+len(rb)-2)) ** 0.5
        if pooled > 0:
            ds.append((sm - bm) / pooled)
    if not ds:
        return float("nan"), float("nan")
    ds.sort()
    lo = ds[int(0.025 * len(ds))]
    hi = ds[int(0.975 * len(ds))]
    return lo, hi


def mann_whitney_u(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Returns (U, normal_approx_p_two_sided). Pure-python, no scipy.

    Used only as a screening signal — for the rebuttal we'll use scipy if
    available; this is a fallback so we don't add a dependency.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    combined = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    combined.sort(key=lambda t: t[0])
    # Average ranks for ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    # Normal approx (no tie correction)
    mu = n1 * n2 / 2.0
    sd = (n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5
    if sd == 0:
        return U, float("nan")
    z = (U - mu) / sd
    # Two-sided p via erf approx
    import math
    p = math.erfc(abs(z) / math.sqrt(2))
    return U, p


# ---------- Cell analysis ----------

COLLAPSE_MIN_CHARS = 200
COLLAPSE_RATIO = 0.25  # variant_len < ratio * orig_len => collapse


def is_collapse(orig_text: str, var_text: str) -> bool:
    return (len(var_text) < COLLAPSE_MIN_CHARS
            or len(var_text) < COLLAPSE_RATIO * max(1, len(orig_text)))


def analyze_cell(model_name: str, variant: str, dataset_maps: dict,
                 model_dir: Path) -> Optional[dict]:
    orig_path = find_variant_file(model_dir, "original")
    var_path = find_variant_file(model_dir, variant)
    if not orig_path or not var_path:
        return None

    orig_by = {p["index"]: p for p in load_problems(orig_path)}
    var_by = {p["index"]: p for p in load_problems(var_path)}

    common = set(orig_by) & set(var_by)
    pairs_stable_drift = []   # (orig_canon, var_canon, problem_type) — non-collapse
    pairs_brittle_drift = []  # non-collapse brittle
    pairs_brittle_collapse = []  # short variant text
    n_stable_collapse = 0  # almost always 0 but tracked for completeness

    for idx in common:
        po, pv = orig_by[idx], var_by[idx]
        if po.get("correct") is not True:
            continue
        var_correct = pv.get("correct")
        if var_correct is None:
            continue
        orig_text = (po.get("solve") or {}).get("solution") or ""
        var_text = (pv.get("solve") or {}).get("solution") or ""
        if not orig_text or not var_text:
            continue
        rmap = dataset_maps.get(idx, {}).get(variant)
        if not rmap:
            continue
        # Canonicalize
        canon_to_ph = {k: f"__V{i}__" for i, k in enumerate(rmap.keys())}
        var_to_ph = {rmap[k]: canon_to_ph[k] for k in rmap}
        orig_canon = canonicalize_text(orig_text, canon_to_ph)
        var_canon = canonicalize_text(var_text, var_to_ph)
        sample = {
            "index": idx,
            "problem_type": po.get("problem_type"),
            "orig_canon": orig_canon,
            "var_canon": var_canon,
            "orig_len": len(orig_text),
            "var_len": len(var_text),
        }
        collapse = is_collapse(orig_text, var_text)
        if var_correct is True:
            if collapse:
                n_stable_collapse += 1
            else:
                pairs_stable_drift.append(sample)
        else:
            if collapse:
                pairs_brittle_collapse.append(sample)
            else:
                pairs_brittle_drift.append(sample)

    if not pairs_stable_drift or not pairs_brittle_drift:
        return None

    metrics = {
        "token_jaccard": metric_token_jaccard,
        "bigram_jaccard": metric_bigram_jaccard,
        "directional_coverage": metric_directional_coverage,
        "window_token_jaccard": metric_window_token_jaccard,
        "window_bigram_jaccard": metric_window_bigram_jaccard,
        "equation_jaccard": metric_equation_jaccard,
    }
    # Headline metric for bootstrap + noise floor (the others stay descriptive)
    HEADLINE = "token_jaccard"

    # Pre-tokenize once per pair to amortize cost (used by token/bigram/window metrics).
    for p in pairs_stable_drift + pairs_brittle_drift:
        p["_otok"] = tokens(p["orig_canon"])
        p["_vtok"] = tokens(p["var_canon"])
        p["_oset"] = set(p["_otok"])
        p["_vset"] = set(p["_vtok"])

    def fast_token_jaccard(p):
        a, b = p["_oset"], p["_vset"]
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def fast_token_jaccard_pair(pa, pb):
        a, b = pa["_oset"], pb["_vset"]
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    out = {
        "model": model_name,
        "variant": variant,
        "n_stable_drift": len(pairs_stable_drift),
        "n_brittle_drift": len(pairs_brittle_drift),
        "n_stable_collapse": n_stable_collapse,
        "n_brittle_collapse": len(pairs_brittle_collapse),
        "brittle_collapse_rate": (len(pairs_brittle_collapse)
                                  / max(1, len(pairs_brittle_collapse) + len(pairs_brittle_drift))),
        "metrics": {},
    }
    # Compute all descriptive metrics (one pass per pair, no bootstrap)
    for mname, mfn in metrics.items():
        s_vals = [mfn(p["orig_canon"], p["var_canon"]) for p in pairs_stable_drift]
        b_vals = [mfn(p["orig_canon"], p["var_canon"]) for p in pairs_brittle_drift]
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
            "delta_mean": sm - bm,
            "cohens_d": d,
            "U": U,
            "p_two_sided": p,
        }

    # Bootstrap + noise floor only on headline metric
    s_vals = [fast_token_jaccard(p) for p in pairs_stable_drift]
    b_vals = [fast_token_jaccard(p) for p in pairs_brittle_drift]
    ci_lo, ci_hi = bootstrap_ci_delta_median(s_vals, b_vals, n_iter=400)
    d_lo, d_hi = bootstrap_ci_cohens_d(s_vals, b_vals, n_iter=400)
    out["metrics"][HEADLINE]["delta_median_ci"] = [ci_lo, ci_hi]
    out["metrics"][HEADLINE]["cohens_d_ci"] = [d_lo, d_hi]

    # Random-pairing noise floor for headline: pair stable orig with random other-problem variant
    import random as _r
    rng = _r.Random(42)
    nf_vals = []
    n = len(pairs_stable_drift)
    if n >= 2:
        for _ in range(min(400, n * (n - 1))):
            i = rng.randrange(n)
            j = rng.randrange(n)
            while j == i:
                j = rng.randrange(n)
            nf_vals.append(fast_token_jaccard_pair(pairs_stable_drift[i],
                                                    pairs_stable_drift[j]))
    out["metrics"][HEADLINE]["noise_floor_median"] = (
        statistics.median(nf_vals) if nf_vals else None)
    out["metrics"][HEADLINE]["noise_floor_mean"] = (
        statistics.fmean(nf_vals) if nf_vals else None)
    out["metrics"][HEADLINE]["noise_floor_n"] = len(nf_vals)
    return out


def main():
    print("Loading dataset rename maps ...", flush=True)
    dataset_maps = load_dataset_maps()
    print(f"  loaded {len(dataset_maps)} problems", flush=True)

    # Multi-cell sweep across all models × 4 surface variants
    # Run all 18 models — non-LLM, fast.
    all_models = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print(f"Models: {all_models}")
    all_results = []

    print(f"\n{'Cell':<46} {'nSd':>4} {'nBd':>4} {'col%':>5} "
          f"{'sMed':>6} {'bMed':>6} {'nfMed':>6} "
          f"{'d':>6} {'d95CI':>14} {'p':>9}")
    print("-" * 122)

    for m in all_models:
        for v in SURFACE_VARIANTS:
            mdir = RESULTS_DIR / m
            if not mdir.exists():
                continue
            res = analyze_cell(m, v, dataset_maps, mdir)
            if res is None:
                continue
            all_results.append(res)
            md = res["metrics"]["token_jaccard"]
            label = f"{m} / {v}"
            ci_lo, ci_hi = md["cohens_d_ci"]
            ci_str = f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"
            print(f"{label:<46} {res['n_stable_drift']:>4} {res['n_brittle_drift']:>4} "
                  f"{res['brittle_collapse_rate']*100:>4.0f}% "
                  f"{md['stable_median']:>6.3f} {md['brittle_median']:>6.3f} "
                  f"{md['noise_floor_median']:>6.3f} "
                  f"{md['cohens_d']:>+6.2f} {ci_str:>14} {md['p_two_sided']:>9.1e}")

    out_path = Path("/home/yurenh2/gap/analysis/structural_overlap_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}  ({len(all_results)} cells)")


if __name__ == "__main__":
    main()
