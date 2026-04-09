"""Three rebuttal figures.

Fig1 — Structural Cohen's d heatmap
       18 models × 5 variants (4 surface + KV).
       Surface cells use the self-anchor metric (model's own original under
       inverse rename). KV uses the canonical-anchor metric.

Fig2 — Rescue rebound rates by variant + condition
       Pooled across 4 models. Bar plot with Wilson 95 % CI.
       Three bars per variant: null / canonical_T2 / own_T2 (KV: only 2).

Fig3 — own_T2 vs canonical_T2 per (model, variant)
       Scatter plot of own_T2 rebound rate vs canonical_T2 rebound rate per
       cell, with the y=x line. Points above the diagonal: own outperforms
       canonical (rare); below: canonical outperforms own (typical).
"""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/yurenh2/gap/analysis")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_LABELS = {
    "descriptive_long": "DL",
    "descriptive_long_confusing": "DLC",
    "descriptive_long_misleading": "DLM",
    "garbled_string": "GS",
    "kernel_variant": "KV",
}
VARIANT_ORDER_SURF = ["descriptive_long", "descriptive_long_confusing",
                      "descriptive_long_misleading", "garbled_string"]
VARIANT_ORDER_ALL = VARIANT_ORDER_SURF + ["kernel_variant"]

# ----------------------------------------------------------------------
# Fig 1 — Structural Cohen's d heatmap
# ----------------------------------------------------------------------

def fig1_structural_d_heatmap():
    """Heatmap of Cohen's d for the stable-vs-brittle structural metric.

    Surface cells: self-anchor (token Jaccard between model's variant
    trajectory and its own original-correct trajectory after canonicalization).
    Source file: structural_overlap_results.json.

    KV cells: canonical-anchor (token Jaccard between model's KV trajectory and
    the dataset's canonical KV solution).
    Source file: kv_overlap_results.json.
    """
    surf = json.load(open(ROOT / "structural_overlap_results.json"))
    kv = json.load(open(ROOT / "kv_overlap_results.json"))

    # Build matrix: rows = models (sorted by mean d), cols = variants (DL, DLC, DLM, GS, KV)
    by_cell = {}
    for c in surf:
        by_cell[(c["model"], c["variant"])] = c["metrics"]["token_jaccard"]["cohens_d"]
    for c in kv:
        by_cell[(c["model"], "kernel_variant")] = c["metrics"]["token_jaccard"]["cohens_d"]

    models = sorted({k[0] for k in by_cell})
    # Sort by mean d across surface variants only (so KV doesn't bias the order)
    def mean_surface_d(m):
        ds = [by_cell.get((m, v)) for v in VARIANT_ORDER_SURF
              if by_cell.get((m, v)) is not None]
        return statistics.fmean(ds) if ds else 0.0
    models.sort(key=mean_surface_d, reverse=True)

    M = np.full((len(models), len(VARIANT_ORDER_ALL)), np.nan)
    for i, m in enumerate(models):
        for j, v in enumerate(VARIANT_ORDER_ALL):
            d = by_cell.get((m, v))
            if d is not None:
                M[i, j] = d

    fig, ax = plt.subplots(figsize=(7, 9))
    vmin = 0.0
    vmax = 1.4
    cmap = plt.cm.viridis
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(VARIANT_ORDER_ALL)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER_ALL])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    # Annotate values
    for i in range(len(models)):
        for j in range(len(VARIANT_ORDER_ALL)):
            v = M[i, j]
            if not math.isnan(v):
                color = "white" if v < 0.7 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color=color)
    # Vertical line separating surface from KV
    ax.axvline(x=3.5, color="white", linewidth=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d (stable − brittle)\non canonicalized token Jaccard",
                   fontsize=9)
    ax.set_title("Structural overlap effect size: stable vs brittle\n"
                 "(surface = self-anchor; KV = canonical-anchor)",
                 fontsize=11)
    ax.set_xlabel("Variant family", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "fig1_structural_d_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ----------------------------------------------------------------------
# Fig 2 — Rescue rebound rates with Wilson CI
# ----------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def fig2_rescue_rates():
    rows = [json.loads(l) for l in open(ROOT / "rescue_results/rescue_30.jsonl")]

    counts = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        counts[(r["variant"], r["condition"])]["n"] += 1
        if r.get("grade") == "CORRECT":
            counts[(r["variant"], r["condition"])]["k"] += 1

    conds_full = ["null", "canonical_T2", "own_T2"]
    cond_color = {"null": "#888888", "canonical_T2": "#1f77b4", "own_T2": "#d62728"}
    cond_label = {"null": "null (generic scaffold)",
                  "canonical_T2": "canonical_T2 (item-specific, expert prose)",
                  "own_T2": "own_T2 (item-specific, model's own work, renamed)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    n_var = len(VARIANT_ORDER_ALL)
    width = 0.27
    x = np.arange(n_var)
    for ci, cond in enumerate(conds_full):
        ks, lows, highs, ps = [], [], [], []
        for v in VARIANT_ORDER_ALL:
            d = counts.get((v, cond))
            if d is None:
                ks.append(0); lows.append(0); highs.append(0); ps.append(0)
                continue
            p, lo, hi = wilson_ci(d["k"], d["n"])
            ps.append(p * 100)
            lows.append((p - lo) * 100)
            highs.append((hi - p) * 100)
            ks.append(d["k"])
        offset = (ci - 1) * width
        ax.bar(x + offset, ps, width=width, color=cond_color[cond], label=cond_label[cond],
               yerr=[lows, highs], capsize=3, error_kw={"elinewidth": 1, "ecolor": "#444444"})
        # Annotate counts above each bar
        for xi, p, k in zip(x + offset, ps, ks):
            if k > 0:
                ax.text(xi, p + 0.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER_ALL], fontsize=10)
    ax.set_ylabel("Rebound rate (%) on flip cases", fontsize=10)
    ax.set_title("Repairability rescue: rebound rate by variant and prefix condition\n"
                 "(pooled across 4 models, n ≈ 100–120 per cell, 95% Wilson CI)",
                 fontsize=11)
    ax.set_ylim(0, 60)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIG_DIR / "fig2_rescue_rebound.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ----------------------------------------------------------------------
# Fig 3 — own_T2 vs canonical_T2 scatter
# ----------------------------------------------------------------------

def fig3_own_vs_canonical_scatter():
    rows = [json.loads(l) for l in open(ROOT / "rescue_results/rescue_30.jsonl")]

    counts = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        counts[(r["model"], r["variant"], r["condition"])]["n"] += 1
        if r.get("grade") == "CORRECT":
            counts[(r["model"], r["variant"], r["condition"])]["k"] += 1

    fig, ax = plt.subplots(figsize=(7, 7))

    models_in_data = sorted({k[0] for k in counts})
    model_color = {
        "claude-sonnet-4":  "#ff7f0e",
        "gemini-2.5-flash": "#2ca02c",
        "gpt-4.1-mini":     "#1f77b4",
        "gpt-4o-mini":      "#d62728",
    }
    var_marker = {
        "descriptive_long": "o",
        "descriptive_long_confusing": "s",
        "descriptive_long_misleading": "^",
        "garbled_string": "D",
    }

    # Diagonal
    ax.plot([0, 0.7], [0, 0.7], "k--", lw=1, alpha=0.5)
    ax.text(0.62, 0.66, "y = x", fontsize=8, alpha=0.6)

    for m in models_in_data:
        for v in VARIANT_ORDER_SURF:
            own = counts.get((m, v, "own_T2"))
            can = counts.get((m, v, "canonical_T2"))
            if own is None or can is None or own["n"] == 0 or can["n"] == 0:
                continue
            x = can["k"] / can["n"]
            y = own["k"] / own["n"]
            ax.scatter(x, y, s=110, c=model_color.get(m, "gray"),
                       marker=var_marker[v], alpha=0.85,
                       edgecolors="black", linewidths=0.6)

    # Build legend
    from matplotlib.lines import Line2D
    model_handles = [Line2D([], [], marker="o", linestyle="", markersize=9,
                            markerfacecolor=c, markeredgecolor="black",
                            markeredgewidth=0.6, label=m)
                     for m, c in model_color.items() if m in models_in_data]
    variant_handles = [Line2D([], [], marker=mk, linestyle="", markersize=9,
                              markerfacecolor="lightgray", markeredgecolor="black",
                              markeredgewidth=0.6, label=VARIANT_LABELS[v])
                       for v, mk in var_marker.items()]
    leg1 = ax.legend(handles=model_handles, loc="upper left", title="Model",
                     fontsize=8, title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, loc="lower right", title="Variant",
              fontsize=8, title_fontsize=9, framealpha=0.95)

    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("canonical_T2 rebound rate", fontsize=10)
    ax.set_ylabel("own_T2 rebound rate", fontsize=10)
    ax.set_title("Per-cell rescue rates: model's own prefix vs canonical prefix\n"
                 "(below diagonal = canonical wins; gpt-4o-mini is the only family above)",
                 fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIG_DIR / "fig3_own_vs_canonical_scatter.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    fig1_structural_d_heatmap()
    fig2_rescue_rates()
    fig3_own_vs_canonical_scatter()
    print("\nAll figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
