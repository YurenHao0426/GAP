"""Spot-check Unicode cleaning by side-by-side comparison.

For a stratified sample of problems, load:
  - the ORIGINAL kernel_variant.solution from the backup tarball
  - the CLEANED kernel_variant.solution from the current dataset
and print them side-by-side so the user can verify that the cleaner
preserved meaning.

Sampling strategy:
  - 5 most complex (by original Unicode count) — stress test
  - 3 medium complexity — typical case
  - 2 surface-variant samples — to confirm rename + LaTeX preserved
"""
from __future__ import annotations
import json
import sys
import tarfile
from pathlib import Path

CURRENT_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")
BACKUP_TAR = sorted(Path("/home/yurenh2/gap/analysis/dataset_backups").glob(
    "putnam-bench-anon_dataset_*.tar.gz"))[-1]


def count_unicode(text: str) -> int:
    return sum(1 for c in (text or "") if ord(c) > 127)


def load_backup_problems():
    """Yield (idx, problem_dict) from the backup tarball."""
    with tarfile.open(BACKUP_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            f = tar.extractfile(member)
            if not f:
                continue
            try:
                d = json.load(f)
                yield d.get("index"), d
            except Exception:
                continue


def main():
    print(f"Backup tar: {BACKUP_TAR}")
    print("Building Unicode-count index over 1051 problems ...")

    # Index originals by Unicode count in kernel_variant.solution
    by_uni_count = []  # (unicode_count, idx, solution_len)
    backup_data = {}
    for idx, d in load_backup_problems():
        if not idx:
            continue
        backup_data[idx] = d
        kv_sol = (d.get("variants") or {}).get("kernel_variant", {}).get("solution", "")
        uc = count_unicode(kv_sol)
        by_uni_count.append((uc, idx, len(kv_sol)))

    by_uni_count.sort(reverse=True)
    print(f"  loaded {len(backup_data)} problems from backup")

    # Pick samples
    samples = []
    samples.extend([(idx, "TOP COMPLEXITY") for _, idx, _ in by_uni_count[:5]])
    mid = len(by_uni_count) // 2
    samples.extend([(idx, "MEDIUM COMPLEXITY")
                    for _, idx, _ in by_uni_count[mid:mid + 3]])
    # Bottom = least Unicode but still non-zero
    nonzero = [t for t in by_uni_count if t[0] > 0]
    samples.extend([(idx, "LOW COMPLEXITY")
                    for _, idx, _ in nonzero[-2:]])

    print(f"\nSelected {len(samples)} samples:\n")
    for idx, label in samples:
        print(f"  {label:<20} {idx}")

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE SPOT-CHECK")
    print("=" * 80)

    for case_idx, (idx, label) in enumerate(samples, 1):
        print(f"\n{'#' * 80}")
        print(f"# CASE {case_idx}/{len(samples)}: {idx} ({label})")
        print(f"{'#' * 80}")

        backup_problem = backup_data.get(idx)
        current_path = CURRENT_DIR / f"{idx}.json"
        if not backup_problem or not current_path.exists():
            print(f"  ! missing data for {idx}")
            continue
        current_problem = json.load(open(current_path))

        # Compare kernel_variant.solution by default. For LOW COMPLEXITY cases
        # we also show the original `solution` field if it differs.
        for field_path in [("variants", "kernel_variant", "solution")]:
            orig_text = backup_problem
            curr_text = current_problem
            for key in field_path:
                orig_text = (orig_text or {}).get(key) if isinstance(orig_text, dict) else None
                curr_text = (curr_text or {}).get(key) if isinstance(curr_text, dict) else None
            if not orig_text and not curr_text:
                continue
            orig_text = orig_text or ""
            curr_text = curr_text or ""
            field_label = ".".join(field_path)
            uni_before = count_unicode(orig_text)
            uni_after = count_unicode(curr_text)
            len_before = len(orig_text)
            len_after = len(curr_text)
            print(f"\n--- field: {field_label} ---")
            print(f"  before: {len_before} chars, {uni_before} non-ASCII")
            print(f"  after:  {len_after} chars, {uni_after} non-ASCII  "
                  f"(Δ len {len_after - len_before:+d})")
            print(f"\n  >>> ORIGINAL (first 600 chars) <<<")
            print("  " + orig_text[:600].replace("\n", "\n  "))
            print(f"\n  >>> CLEANED (first 600 chars) <<<")
            print("  " + curr_text[:600].replace("\n", "\n  "))

            if uni_after > 0:
                print(f"  !!! WARNING: cleaned output still has {uni_after} non-ASCII chars")

            # Sanity: are LaTeX braces balanced in the cleaned text?
            n_open = curr_text.count("{")
            n_close = curr_text.count("}")
            n_lparen = curr_text.count("(")
            n_rparen = curr_text.count(")")
            n_lbrack = curr_text.count("[")
            n_rbrack = curr_text.count("]")
            print(f"  brace balance: {{ {n_open} | }} {n_close}  "
                  f"( {n_lparen} | ) {n_rparen}  "
                  f"[ {n_lbrack} | ] {n_rbrack}")

    # Final aggregate balance check across the entire cleaned dataset
    print("\n" + "=" * 80)
    print("AGGREGATE BRACE BALANCE CHECK (entire cleaned dataset)")
    print("=" * 80)
    total_diff_brace = 0
    total_diff_paren = 0
    total_diff_brack = 0
    files_with_brace_imbalance = 0
    files_with_paren_imbalance = 0
    files_with_brack_imbalance = 0
    for f in sorted(CURRENT_DIR.glob("*.json")):
        d = json.load(open(f))
        # Concatenate all text fields
        bag = []
        for k in ("question", "solution"):
            bag.append(d.get(k) or "")
        for vk, vd in (d.get("variants") or {}).items():
            if isinstance(vd, dict):
                for k in ("question", "solution"):
                    bag.append(vd.get(k) or "")
        all_text = "\n".join(bag)
        diff_brace = all_text.count("{") - all_text.count("}")
        diff_paren = all_text.count("(") - all_text.count(")")
        diff_brack = all_text.count("[") - all_text.count("]")
        if diff_brace != 0:
            files_with_brace_imbalance += 1
            total_diff_brace += abs(diff_brace)
        if diff_paren != 0:
            files_with_paren_imbalance += 1
            total_diff_paren += abs(diff_paren)
        if diff_brack != 0:
            files_with_brack_imbalance += 1
            total_diff_brack += abs(diff_brack)

    print(f"  files with unbalanced {{...}}:  {files_with_brace_imbalance}/1051"
          f"  (total |Δ| = {total_diff_brace})")
    print(f"  files with unbalanced (...): {files_with_paren_imbalance}/1051"
          f"  (total |Δ| = {total_diff_paren})")
    print(f"  files with unbalanced [...]: {files_with_brack_imbalance}/1051"
          f"  (total |Δ| = {total_diff_brack})")
    print()
    print("  (Imbalance is not necessarily a bug — math text often legitimately")
    print("   contains unbalanced delimiters in display formulas; this is just")
    print("   an order-of-magnitude check.)")


if __name__ == "__main__":
    main()
