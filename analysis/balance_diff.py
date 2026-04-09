"""Compare brace/paren/bracket balance BEFORE vs AFTER cleaning to check
whether the cleaner introduced any new imbalance."""
from __future__ import annotations
import json
import tarfile
from pathlib import Path
from collections import Counter

CURRENT_DIR = Path("/home/yurenh2/gap/putnam-bench-anon/dataset")
BACKUP_TAR = sorted(Path("/home/yurenh2/gap/analysis/dataset_backups").glob(
    "putnam-bench-anon_dataset_*.tar.gz"))[-1]


def all_text(d: dict) -> str:
    out = []
    for k in ("question", "solution"):
        out.append(d.get(k) or "")
    for vk, vd in (d.get("variants") or {}).items():
        if isinstance(vd, dict):
            for k in ("question", "solution"):
                out.append(vd.get(k) or "")
    return "\n".join(out)


def balance(text: str):
    return (
        text.count("{") - text.count("}"),
        text.count("(") - text.count(")"),
        text.count("[") - text.count("]"),
    )


def main():
    print("Loading backup ...")
    backup = {}
    with tarfile.open(BACKUP_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            f = tar.extractfile(member)
            if not f:
                continue
            d = json.load(f)
            backup[d.get("index")] = all_text(d)
    print(f"  loaded {len(backup)} backup problems")

    print("Loading current ...")
    current = {}
    for f in sorted(CURRENT_DIR.glob("*.json")):
        d = json.load(open(f))
        current[d.get("index")] = all_text(d)
    print(f"  loaded {len(current)} current problems")

    # Per-file balance diff
    introduced_imbalance = []
    fixed_imbalance = []
    same_imbalance = 0
    same_balanced = 0

    n_brace_changed = 0
    n_paren_changed = 0
    n_brack_changed = 0

    for idx in sorted(backup):
        b_before = balance(backup[idx])
        b_after = balance(current.get(idx, ""))
        was_bal = b_before == (0, 0, 0)
        is_bal = b_after == (0, 0, 0)
        if b_before != b_after:
            if was_bal and not is_bal:
                introduced_imbalance.append((idx, b_before, b_after))
            elif not was_bal and is_bal:
                fixed_imbalance.append((idx, b_before, b_after))
        else:
            if is_bal:
                same_balanced += 1
            else:
                same_imbalance += 1
        if b_before[0] != b_after[0]: n_brace_changed += 1
        if b_before[1] != b_after[1]: n_paren_changed += 1
        if b_before[2] != b_after[2]: n_brack_changed += 1

    print(f"\n=== Per-file balance change summary ===")
    print(f"  Files with no change in any balance:")
    print(f"    balanced both before and after: {same_balanced}")
    print(f"    imbalanced before and after (same imbalance): {same_imbalance}")
    print(f"  Files where cleaner INTRODUCED new imbalance: "
          f"{len(introduced_imbalance)}")
    print(f"  Files where cleaner FIXED prior imbalance: {len(fixed_imbalance)}")
    print()
    print(f"  Files where {{ balance changed: {n_brace_changed}")
    print(f"  Files where ( balance changed: {n_paren_changed}")
    print(f"  Files where [ balance changed: {n_brack_changed}")

    if introduced_imbalance:
        print(f"\n!!! Cleaner-introduced imbalances ({len(introduced_imbalance)}):")
        for idx, before, after in introduced_imbalance[:10]:
            print(f"    {idx}: before={before}, after={after}")
    else:
        print("\n  ✓ No cleaner-introduced imbalances found.")

    if fixed_imbalance:
        print(f"\n  Cleaner-fixed imbalances (top 10):")
        for idx, before, after in fixed_imbalance[:10]:
            print(f"    {idx}: before={before}, after={after}")


if __name__ == "__main__":
    main()
