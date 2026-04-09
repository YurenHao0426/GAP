"""Self-correction / metacognition probe.

Scan model trajectories for self-correction markers and compute:
1. Attempt rate (trajectory contains a self-correction marker) per (model, variant, group)
2. Whether self-correction attempt rate differs between stable / brittle-drift / rescued cases
3. Conditional success: among trajectories with a self-correction attempt, what fraction is graded CORRECT?

Self-correction markers (case-insensitive, word-boundary):
- "wait" (e.g., "Wait, let me reconsider")
- "actually" (e.g., "Actually, I think...")
- "let me reconsider"
- "let me redo"
- "let me try again"
- "I made a mistake"
- "this is wrong"
- "on second thought"
- "correction:"
- "scratch that"
- "I was wrong"
- "let me start over"

Uses three data sources:
A. The original 18-model results in /home/yurenh2/gap/results_new/ (stable + brittle drift + collapse)
B. The rescue trajectories in analysis/rescue_results/rescue_30.jsonl (3 conditions × 4 models × 5 variants)
"""
from __future__ import annotations
import json
import re
import os
import sys
import statistics
from pathlib import Path
from collections import defaultdict, Counter

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from structural_overlap import find_variant_file, load_problems, RESULTS_DIR, SURFACE_VARIANTS

SC_PATTERNS = [
    re.compile(r"\bwait\b[,.]?\s+(let|actually|that|i)", re.IGNORECASE),
    re.compile(r"\bactually[,.]\s", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+reconsider", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+redo", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+try\s+(this\s+)?again", re.IGNORECASE),
    re.compile(r"\bi\s+made\s+a\s+mistake", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+(wrong|incorrect)", re.IGNORECASE),
    re.compile(r"\bon\s+second\s+thought", re.IGNORECASE),
    re.compile(r"\bcorrection[:\s]", re.IGNORECASE),
    re.compile(r"\bscratch\s+that", re.IGNORECASE),
    re.compile(r"\bi\s+was\s+wrong", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+start\s+over", re.IGNORECASE),
    re.compile(r"\bhmm[,.]\s+(actually|wait|that)", re.IGNORECASE),
    re.compile(r"\bi\s+need\s+to\s+(redo|reconsider)", re.IGNORECASE),
    re.compile(r"\boh\s+wait", re.IGNORECASE),
]


def has_self_correction(text: str) -> bool:
    if not text:
        return False
    for pat in SC_PATTERNS:
        if pat.search(text):
            return True
    return False


def count_sc_markers(text: str) -> int:
    if not text:
        return 0
    return sum(len(pat.findall(text)) for pat in SC_PATTERNS)


# ---------- Source A: 18-model original results ----------

def analyze_18_models():
    """Self-correction rates in original solver runs across all 18 models."""
    base = RESULTS_DIR
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])
    print(f"\n=== SELF-CORRECTION IN 18-MODEL ORIGINAL RUNS ===\n")
    print(f"Markers used: {len(SC_PATTERNS)} regex patterns")
    print(f"Definition: trajectory contains at least one match.\n")

    rows = []
    for m in models:
        mdir = base / m
        for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
            vp = find_variant_file(mdir, v)
            if not vp:
                continue
            problems = load_problems(vp)
            n_total = 0
            n_sc = 0
            n_correct_sc = 0
            n_correct_total = 0
            n_wrong_sc = 0
            n_wrong_total = 0
            for p in problems:
                text = (p.get("solve") or {}).get("solution") or ""
                if not text:
                    continue
                correct = p.get("correct")
                if correct is None:
                    continue
                n_total += 1
                sc = has_self_correction(text)
                if sc: n_sc += 1
                if correct is True:
                    n_correct_total += 1
                    if sc: n_correct_sc += 1
                else:
                    n_wrong_total += 1
                    if sc: n_wrong_sc += 1
            if n_total > 0:
                rows.append({
                    "model": m, "variant": v, "n": n_total,
                    "sc_rate": n_sc / n_total,
                    "n_correct": n_correct_total,
                    "n_correct_sc_rate": n_correct_sc / max(1, n_correct_total),
                    "n_wrong": n_wrong_total,
                    "n_wrong_sc_rate": n_wrong_sc / max(1, n_wrong_total),
                })

    # Print compact table: per (variant) average across models
    print(f"{'Variant':<24} {'mean SC%':>10} {'SC%|correct':>14} {'SC%|wrong':>12} {'asym (wrong-correct)':>22}")
    print("-" * 90)
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    for v in ["original"] + SURFACE_VARIANTS + ["kernel_variant"]:
        rs = by_var.get(v, [])
        if not rs:
            continue
        m_sc = statistics.fmean(r["sc_rate"] for r in rs) * 100
        m_sc_c = statistics.fmean(r["n_correct_sc_rate"] for r in rs) * 100
        m_sc_w = statistics.fmean(r["n_wrong_sc_rate"] for r in rs) * 100
        asym = m_sc_w - m_sc_c
        print(f"{v:<24} {m_sc:>9.1f}% {m_sc_c:>13.1f}% {m_sc_w:>11.1f}% {asym:>+21.1f}pp")

    # Per-model leader board
    print(f"\n{'Model':<22} {'mean SC% (all variants)':>26}")
    print("-" * 50)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r["sc_rate"])
    model_avgs = sorted([(m, statistics.fmean(vs) * 100) for m, vs in by_model.items()],
                        key=lambda t: -t[1])
    for m, avg in model_avgs:
        print(f"{m:<22} {avg:>25.1f}%")

    return rows


# ---------- Source B: rescue trajectories ----------

def analyze_rescue():
    path = THIS_DIR / "rescue_results/rescue_30.jsonl"
    rows = [json.loads(l) for l in open(path)]
    print(f"\n\n=== SELF-CORRECTION IN 1{{,}}529 RESCUE TRAJECTORIES ===\n")

    # Group by (model, variant, condition, grade)
    counts = defaultdict(lambda: {"n": 0, "sc": 0})
    for r in rows:
        text = r.get("student_solution") or ""
        if not text:
            continue
        key = (r["model"], r["variant"], r["condition"], r.get("grade"))
        counts[key]["n"] += 1
        if has_self_correction(text):
            counts[key]["sc"] += 1

    # Aggregate per (variant, condition, grade)
    by_vcg = defaultdict(lambda: {"n": 0, "sc": 0})
    for k, d in counts.items():
        m, v, c, g = k
        by_vcg[(v, c, g)]["n"] += d["n"]
        by_vcg[(v, c, g)]["sc"] += d["sc"]

    print(f"{'Variant':<24} {'Condition':<14} {'CORRECT-SC%':>14} {'INCORRECT-SC%':>16}")
    print("-" * 80)
    for v in ["descriptive_long","descriptive_long_confusing","descriptive_long_misleading","garbled_string","kernel_variant"]:
        for c in ["null", "canonical_T2", "own_T2"]:
            cor = by_vcg.get((v, c, "CORRECT"), {"n": 0, "sc": 0})
            inc = by_vcg.get((v, c, "INCORRECT"), {"n": 0, "sc": 0})
            if cor["n"] == 0 and inc["n"] == 0:
                continue
            sc_c = cor["sc"] / max(1, cor["n"]) * 100 if cor["n"] else 0
            sc_i = inc["sc"] / max(1, inc["n"]) * 100 if inc["n"] else 0
            print(f"{v:<24} {c:<14} {sc_c:>11.1f}% (n={cor['n']:>3}) {sc_i:>13.1f}% (n={inc['n']:>3})")
        print()

    return counts


def main():
    rows_18 = analyze_18_models()
    json.dump(rows_18, open(THIS_DIR / "self_correction_18models.json", "w"), indent=2)
    counts_rescue = analyze_rescue()
    print("\nSaved -> analysis/self_correction_18models.json")


if __name__ == "__main__":
    main()
