"""Unicode audit for PutnamGAP dataset.

Scans all JSON files in the dataset, finds all non-ASCII characters in text
fields (question, solution across all variants), and reports:

1. How many files contain Unicode
2. Top Unicode characters by total frequency with suggested LaTeX replacements
3. Which fields are most affected
4. Per-file tallies
5. Samples of lines showing each unusual character in context
6. A machine-readable JSON report for downstream cleaning

Does NOT modify any file. Read-only audit.
"""
from __future__ import annotations
import json
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

# Both copies of the dataset
DIRS = [
    Path("/home/yurenh2/gap/putnam-bench-anon/dataset"),
    Path("/home/yurenh2/gap/putnamsup/PutnamGAP"),
]

# Text-bearing fields we care about
TOP_LEVEL_TEXT_FIELDS = ["question", "solution"]
VARIANT_TEXT_FIELDS = ["question", "solution"]
VARIANT_KEYS = [
    "descriptive_long",
    "descriptive_long_confusing",
    "descriptive_long_misleading",
    "garbled_string",
    "kernel_variant",
    "original_kernel_variant",
]

# Suggested LaTeX replacements for common math Unicode. (Informational — the
# audit does not apply these.) Each entry is (unicode_char, latex_suggestion).
SUGGESTED_LATEX = {
    # Greek lower case
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega",
    # Greek upper case
    "Α": "A", "Β": "B", "Γ": r"\Gamma", "Δ": r"\Delta", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi",
    "Ω": r"\Omega",
    # Math operators & relations
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "±": r"\pm", "∓": r"\mp", "×": r"\times",
    "÷": r"\div", "·": r"\cdot", "∙": r"\cdot",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "∆": r"\Delta",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "√": r"\sqrt{}",
    "∮": r"\oint", "∴": r"\therefore", "∵": r"\because",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "⊃": r"\supset", "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap",
    "∧": r"\land", "∨": r"\lor", "¬": r"\neg",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "⟨": r"\langle", "⟩": r"\rangle", "⌊": r"\lfloor", "⌋": r"\rfloor",
    "⌈": r"\lceil", "⌉": r"\rceil",
    "∅": r"\emptyset", "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}",
    "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    # Subscripts / superscripts (common ones only)
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5",
    "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "ₐ": "_a", "ᵢ": "_i", "ⱼ": "_j", "ₖ": "_k", "ₙ": "_n",
    # Fractions
    "½": r"\frac{1}{2}", "⅓": r"\frac{1}{3}", "⅔": r"\frac{2}{3}",
    "¼": r"\frac{1}{4}", "¾": r"\frac{3}{4}",
    # Punctuation / whitespace
    "—": "---", "–": "--", "…": r"\ldots",
    "‘": "`", "’": "'", "“": "``", "”": "''",
    "°": r"^\circ",
    "\u00A0": " (nbsp)",  # non-breaking space
    "\u2009": " (thin space)",
    "\u200b": " (zero-width space)",
    "\u2026": r"\ldots",
    "\u2212": "-",  # Unicode minus vs hyphen
}


def is_non_ascii(ch: str) -> bool:
    return ord(ch) > 127


def extract_text_fields(problem: dict):
    """Yield (field_path, text) for every text-bearing field in a problem."""
    idx = problem.get("index", "?")
    for k in TOP_LEVEL_TEXT_FIELDS:
        v = problem.get(k)
        if isinstance(v, str):
            yield f"{idx}:{k}", v
    for vk in VARIANT_KEYS:
        vd = (problem.get("variants") or {}).get(vk)
        if not isinstance(vd, dict):
            continue
        for k in VARIANT_TEXT_FIELDS:
            v = vd.get(k)
            if isinstance(v, str):
                yield f"{idx}:variants.{vk}.{k}", v


def audit_dir(dataset_dir: Path, label: str):
    print(f"\n{'=' * 76}")
    print(f"Auditing {label}: {dataset_dir}")
    print(f"{'=' * 76}")

    files = sorted(dataset_dir.glob("*.json"))
    print(f"Files: {len(files)}")

    char_counter = Counter()                # unicode char -> total occurrences
    field_char_counter = defaultdict(Counter)  # field_name -> Counter
    files_with_unicode = set()              # set of problem indices
    per_field_counts = Counter()            # {question, solution, variants.DL.question, ...} -> n files with unicode
    examples = defaultdict(list)            # char -> list of (context, path)
    total_chars = 0
    total_unicode = 0

    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  ! {f.name}: JSON parse error: {e}")
            continue
        file_had_unicode = False
        for path, text in extract_text_fields(d):
            if not text:
                continue
            total_chars += len(text)
            nas = [c for c in text if is_non_ascii(c)]
            if not nas:
                continue
            file_had_unicode = True
            total_unicode += len(nas)
            # tally
            for c in nas:
                char_counter[c] += 1
                # short field label (strip problem index prefix)
                short = path.split(":", 1)[1]
                field_char_counter[short][c] += 1
                per_field_counts[short] += 1
                # collect up to 3 examples per char with ±20 char context
                if len(examples[c]) < 3:
                    idx = text.find(c)
                    start = max(0, idx - 25)
                    end = min(len(text), idx + 25)
                    ctx = text[start:end].replace("\n", " ")
                    examples[c].append((ctx, path))
        if file_had_unicode:
            files_with_unicode.add(d.get("index", f.name))

    # Report
    print(f"\nTotal characters scanned: {total_chars:,}")
    print(f"Non-ASCII characters: {total_unicode:,} ({total_unicode/total_chars*100:.2f}%)")
    print(f"Files with any Unicode: {len(files_with_unicode)}/{len(files)} "
          f"({len(files_with_unicode)/len(files)*100:.1f}%)")
    print(f"Distinct Unicode code points: {len(char_counter)}")

    print(f"\n--- Top 40 Unicode characters by frequency ---")
    print(f"{'char':<6} {'hex':<8} {'count':>8}  name / suggested LaTeX")
    print("-" * 76)
    for c, n in char_counter.most_common(40):
        name = unicodedata.name(c, "?")
        hex_val = f"U+{ord(c):04X}"
        suggestion = SUGGESTED_LATEX.get(c, "")
        display_c = c if c.isprintable() and ord(c) > 0x20 else repr(c)
        print(f"{display_c:<6} {hex_val:<8} {n:>8}  {name[:45]:<45} {suggestion}")

    # Per-field breakdown
    print(f"\n--- Unicode per field (top 15 fields with most Unicode) ---")
    print(f"{'field':<50} {'total unicode':>15}")
    print("-" * 70)
    for field, cnt in Counter({f: sum(c.values()) for f, c in field_char_counter.items()}).most_common(15):
        print(f"{field:<50} {cnt:>15}")

    # Examples for top 10 chars
    print(f"\n--- Example contexts for top 10 Unicode chars ---")
    for c, n in char_counter.most_common(10):
        name = unicodedata.name(c, "?")
        display_c = c if c.isprintable() and ord(c) > 0x20 else repr(c)
        print(f"\n  {display_c} (U+{ord(c):04X}, {name}, n={n}):")
        for ctx, path in examples[c][:2]:
            print(f"    [{path}]")
            print(f"      …{ctx}…")

    # Machine-readable summary
    summary = {
        "dataset_dir": str(dataset_dir),
        "n_files": len(files),
        "n_files_with_unicode": len(files_with_unicode),
        "pct_files_with_unicode": 100 * len(files_with_unicode) / max(1, len(files)),
        "total_chars": total_chars,
        "total_unicode": total_unicode,
        "distinct_codepoints": len(char_counter),
        "top_chars": [
            {"char": c, "codepoint": f"U+{ord(c):04X}",
             "name": unicodedata.name(c, "?"),
             "count": n,
             "suggested_latex": SUGGESTED_LATEX.get(c, ""),
             "examples": [{"path": path, "context": ctx}
                          for ctx, path in examples[c][:3]]}
            for c, n in char_counter.most_common(80)
        ],
        "per_field_unicode_counts": dict(
            Counter({f: sum(c.values()) for f, c in field_char_counter.items()})
            .most_common(30)),
        "files_with_unicode_indices": sorted(files_with_unicode),
    }
    return summary


def main():
    all_summaries = []
    for d in DIRS:
        if d.exists():
            s = audit_dir(d, d.name)
            s["label"] = d.name
            all_summaries.append(s)
        else:
            print(f"  (skipping missing dir {d})")

    out_path = Path("/home/yurenh2/gap/analysis/unicode_audit.json")
    json.dump(all_summaries, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n\nSaved machine-readable summary -> {out_path}")


if __name__ == "__main__":
    main()
