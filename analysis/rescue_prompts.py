"""Rescue-experiment prompt construction.

For each (model, variant, flip-case) we build prompts under three conditions:
- own_T2:        model's own original-correct trajectory truncated at first
                 formal equation (with leakage filter), variables auto-renamed
                 to variant names via the dataset's rename map
- canonical_T2:  the dataset's canonical variant solution truncated at first
                 formal equation (no rename needed; already in variant naming)
- null:          generic content-free scaffold

Truncation rule (event-boundary):
  1. Find the FIRST display-math block ($$...$$, \\[...\\], \\begin{equation/align/...})
  2. If none, fall back to the first line containing a substantive math relation
     (>=, <=, =, <, >, ≡, ∈) that is not merely a definition (e.g., 'let x:=...')
  3. The T2 prefix INCLUDES that first formal relation
  4. Apply leakage filter BEFORE returning: stop at the earliest of:
     - any line containing \\boxed
     - any line containing 'therefore', 'hence', 'we conclude', 'the answer',
       'we obtain', 'thus', 'it suffices', 'we have proved', 'as a result'
     - any line containing the dataset's recorded final_answer string
"""
from __future__ import annotations
import re
from typing import Optional, Dict


# ---------- Display-math detection ----------

# Order matters: try richest patterns first
_DISPLAY_MATH_PATTERNS = [
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\\begin\{equation\*?\}.+?\\end\{equation\*?\}", re.DOTALL),
    re.compile(r"\\begin\{align\*?\}.+?\\end\{align\*?\}", re.DOTALL),
    re.compile(r"\\begin\{gather\*?\}.+?\\end\{gather\*?\}", re.DOTALL),
    re.compile(r"\\begin\{eqnarray\*?\}.+?\\end\{eqnarray\*?\}", re.DOTALL),
]


def _first_display_math_end(text: str) -> Optional[int]:
    """Return the end position of the first display-math block, or None."""
    earliest = None
    for pat in _DISPLAY_MATH_PATTERNS:
        m = pat.search(text)
        if m:
            if earliest is None or m.end() < earliest:
                earliest = m.end()
    return earliest


# Inline relation fallback: first line with a "real" relation
_INLINE_REL_RE = re.compile(
    r"[A-Za-z\)\]\}\d_]\s*(?:=|<|>|\\le[q]?|\\ge[q]?|\\equiv|\\in)\s*[A-Za-z\(\[\{\d\\\-]"
)
# Definition exclusion: lines that are 'let x = ...' or 'denote ...' are setup,
# not actual derivations. We allow them in the prefix but don't stop on them.
_DEFINITION_RE = re.compile(
    r"^\s*(?:let|denote|define|set|put|call|consider|introduce|let us)\b",
    re.IGNORECASE
)


def _first_inline_relation_line_end(text: str) -> Optional[int]:
    """Find the end of the first line containing a non-definition math relation.

    Returns absolute character offset (one past the newline)."""
    pos = 0
    while pos < len(text):
        nl = text.find("\n", pos)
        line_end = nl if nl != -1 else len(text)
        line = text[pos:line_end]
        if _INLINE_REL_RE.search(line) and not _DEFINITION_RE.search(line):
            return line_end + 1 if nl != -1 else line_end
        pos = line_end + 1
        if nl == -1:
            break
    return None


# ---------- Leakage detection ----------

LEAKAGE_PATTERNS = [
    re.compile(r"\\boxed\b", re.IGNORECASE),
    re.compile(r"\btherefore\b", re.IGNORECASE),
    re.compile(r"\bhence\b", re.IGNORECASE),
    re.compile(r"\bwe conclude\b", re.IGNORECASE),
    re.compile(r"\bthe answer\b", re.IGNORECASE),
    re.compile(r"\bwe obtain\b", re.IGNORECASE),
    re.compile(r"\bthus\b", re.IGNORECASE),
    re.compile(r"\bit suffices\b", re.IGNORECASE),
    re.compile(r"\bwe have proved\b", re.IGNORECASE),
    re.compile(r"\bwe have shown\b", re.IGNORECASE),
    re.compile(r"\bas a result\b", re.IGNORECASE),
    re.compile(r"\bin conclusion\b", re.IGNORECASE),
    re.compile(r"\bthe final answer\b", re.IGNORECASE),
    re.compile(r"\bso the answer\b", re.IGNORECASE),
]


def _first_leakage_pos(text: str, final_answer: Optional[str] = None) -> Optional[int]:
    """Return the starting char position of the earliest leakage marker."""
    earliest = None
    for pat in LEAKAGE_PATTERNS:
        m = pat.search(text)
        if m:
            if earliest is None or m.start() < earliest:
                earliest = m.start()
    if final_answer:
        # Final-answer leakage: only check if the answer string is non-trivial
        fa = final_answer.strip()
        if 8 <= len(fa) <= 200:
            idx = text.find(fa)
            if idx != -1:
                if earliest is None or idx < earliest:
                    earliest = idx
    return earliest


# ---------- T2 truncation ----------

MIN_PREFIX_CHARS = 50
MAX_PREFIX_CHARS = 2400  # roughly 600 tokens


def truncate_T2(text: str, final_answer: Optional[str] = None) -> Optional[str]:
    """Return the T2 (after-first-equation) prefix, or None if not detectable.

    T2 = up to and including the first formal equation, then capped by leakage
    filter and MAX_PREFIX_CHARS.
    """
    if not text:
        return None
    end = _first_display_math_end(text)
    if end is None:
        end = _first_inline_relation_line_end(text)
    if end is None:
        return None
    prefix = text[:end]
    # Apply leakage filter BEFORE the equation if a leakage marker appears earlier
    leak = _first_leakage_pos(prefix, final_answer)
    if leak is not None and leak < end:
        prefix = text[:leak].rstrip()
    # Cap length
    if len(prefix) > MAX_PREFIX_CHARS:
        prefix = prefix[:MAX_PREFIX_CHARS]
        # Trim at last newline to avoid cutting mid-sentence
        last_nl = prefix.rfind("\n")
        if last_nl > MIN_PREFIX_CHARS:
            prefix = prefix[:last_nl]
    if len(prefix) < MIN_PREFIX_CHARS:
        return None
    return prefix.rstrip()


# ---------- Variable rename for own prefix ----------

def rename_own_prefix(prefix: str, rename_map: Dict[str, str]) -> str:
    """Apply orig->variant rename mapping to the model's own prefix.

    Sort longest-first to avoid prefix collisions (e.g., 'al' eating 'almondtree').
    Use word-boundary regex. Pass replacement via lambda to avoid escape-sequence
    interpretation when the variant name starts with '\\x', '\\g', etc.
    """
    if not prefix or not rename_map:
        return prefix
    items = sorted(rename_map.items(), key=lambda kv: -len(kv[0]))
    out = prefix
    for src, dst in items:
        if not src:
            continue
        pat = r"(?<![A-Za-z0-9_])" + re.escape(src) + r"(?![A-Za-z0-9_])"
        # Use a lambda so dst is treated literally (no \1, \x, etc. escapes).
        out = re.sub(pat, lambda _m, _dst=dst: _dst, out)
    return out


# ---------- Null scaffold ----------

NULL_SCAFFOLD = (
    "Let us proceed carefully. We will first identify the relevant variables "
    "and their roles, then state the governing relations of the problem, and "
    "finally develop the argument step by step."
)


# ---------- Prompt builders ----------

# We tell the model to PRODUCE the complete solution that begins with the
# provided prefix verbatim. This means the grader will see one continuous
# solution that starts with the injected setup. The instruction to begin
# verbatim avoids the model paraphrasing the prefix and removing the very
# representational anchor we are testing.

RESCUE_USER_TEMPLATE = """Please solve the following mathematical problem.

PROBLEM:
{problem_statement}

You must structure your solution as a continuation of the partial work below.
Begin your solution with the partial work copied verbatim, then continue
seamlessly to a complete answer.

PARTIAL WORK (to copy verbatim at the start of your solution):
{prefix}

Provide a complete, rigorous solution. Return your response in JSON format:
{{"solution": "your complete solution starting with the partial work above and continuing to the end",
  "final_answer": "your final answer in clear, concise form"}}"""


NULL_USER_TEMPLATE = """Please solve the following mathematical problem.

PROBLEM:
{problem_statement}

{scaffold}

Provide a complete, rigorous solution. Return your response in JSON format:
{{"solution": "your complete step-by-step solution",
  "final_answer": "your final answer in clear, concise form"}}"""


def build_rescue_prompt(problem_statement: str, prefix: str) -> str:
    return RESCUE_USER_TEMPLATE.format(
        problem_statement=problem_statement, prefix=prefix)


def build_null_prompt(problem_statement: str) -> str:
    return NULL_USER_TEMPLATE.format(
        problem_statement=problem_statement, scaffold=NULL_SCAFFOLD)


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Quick smoke test on a real flip case
    import json
    import sys
    sys.path.insert(0, "/home/yurenh2/gap/analysis")
    from structural_overlap import find_variant_file, load_problems

    # Pick gpt-4.1-mini original on a known problem
    op = find_variant_file(
        __import__("pathlib").Path("/home/yurenh2/gap/results_new/gpt-4.1-mini"),
        "original")
    probs = {p["index"]: p for p in load_problems(op)}
    sample = next(p for idx, p in probs.items()
                  if p.get("correct") is True and (p.get("solve") or {}).get("solution"))
    text = sample["solve"]["solution"]
    fa = sample["solve"].get("final_answer")
    print(f"Sample index: {sample['index']}, type: {sample['problem_type']}")
    print(f"Original solution length: {len(text)} chars")
    print(f"Recorded final_answer: {fa[:200] if fa else None!r}")
    pre = truncate_T2(text, fa)
    print(f"\n--- T2 PREFIX ({len(pre or '')} chars) ---")
    print(pre)
    print("--- END ---")

    # Test rename: load 1987-B-2 dataset to get a sample map
    ds = json.load(open("/home/yurenh2/gap/putnam-bench-anon/dataset/1987-B-2.json"))
    rmap_raw = ds["variants"]["garbled_string"]["map"]
    rmap = (eval(rmap_raw, {"__builtins__": {}}, {})
            if isinstance(rmap_raw, str) else rmap_raw)
    print(f"\nRename map: {rmap}")
    test_text = "Let n be a positive integer and let f be a continuous function. Then $f(n) = 0$."
    print(f"\nOriginal: {test_text}")
    print(f"Renamed:  {rename_own_prefix(test_text, rmap)}")
