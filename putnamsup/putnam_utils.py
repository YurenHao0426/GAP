import os
import json
from typing import Dict, Any, Generator, Tuple, Optional, List

# Supported variants as seen in putnamgap_viewer.py
SUPPORTED_VARIANTS = [
    "original",
    "descriptive_long",
    "descriptive_long_confusing",
    "descriptive_long_misleading",
    "garbled_string",
    "kernel_variant",
]

def get_original_qa(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract original question and solution."""
    question = d.get("question")
    solution = d.get("solution", d.get("answer"))
    return question, solution

def get_variant_qa(d: Dict[str, Any], variant_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract variant question and solution."""
    variants = d.get("variants")
    if not isinstance(variants, dict):
        return None, None
    var = variants.get(variant_key)
    if not isinstance(var, dict):
        return None, None
    question = var.get("question")
    solution = var.get("solution", var.get("answer"))
    return question, solution

def load_dataset(data_dir: str, selected_variants: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Iterates over all JSON files in data_dir and yields problem instances.
    Each instance is a dict with keys: file_index, type, variant, question, solution.
    
    Args:
        data_dir: Path to the dataset directory.
        selected_variants: List of variants to include. If None, include all.
                           Supported values are in SUPPORTED_VARIANTS.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    # Validate selected_variants
    if selected_variants:
        for v in selected_variants:
            if v not in SUPPORTED_VARIANTS:
                print(f"Warning: Variant '{v}' not recognized. Supported: {SUPPORTED_VARIANTS}")
    
    # If no filter provided, use all supported
    target_variants = selected_variants if selected_variants else SUPPORTED_VARIANTS

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".json")]
    files.sort()

    for f in files:
        filepath = os.path.join(data_dir, f)
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        file_index = data.get("index", f) # Use filename as index if 'index' key missing
        prob_type = data.get("problem_type", "unknown")

        # 1. Original
        if "original" in target_variants:
            q, a = get_original_qa(data)
            if q and a:
                yield {
                    "file_index": file_index,
                    "problem_type": prob_type,
                    "variant": "original",
                    "question": q,
                    "solution": a
                }

        # 2. Variants
        for var_key in SUPPORTED_VARIANTS:
            if var_key == "original": continue
            if var_key not in target_variants: continue
            
            q, a = get_variant_qa(data, var_key)
            if q and a:
                yield {
                    "file_index": file_index,
                    "problem_type": prob_type,
                    "variant": var_key,
                    "question": q,
                    "solution": a
                }
