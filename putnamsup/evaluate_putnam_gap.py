import json
import argparse
import re

def normalize_answer(text):
    """Simple normalization for comparison."""
    if text is None: return ""
    text = text.strip().lower()
    # Remove latex formatting for simple check
    text = re.sub(r'\\[\(\)\[\]]', ' ', text)
    return text

def simple_evaluate(ground_truth, generated):
    """
    A very naive evaluator. 
    Returns True if the generated answer seems to contain the ground truth 
    (if ground truth is short) or based on some heuristics.
    """
    gt_norm = normalize_answer(ground_truth)
    gen_norm = normalize_answer(generated)
    
    # If ground truth is very short (likely a number or variable), check if it's in the generated text
    if len(gt_norm) < 20:
        return gt_norm in gen_norm
    
    # For longer proofs, this metric is useless.
    return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate PutnamGAP results")
    parser.add_argument("--results_file", type=str, required=True, help="Path to JSONL results file")
    args = parser.parse_args()

    total = 0
    correct_heuristic = 0
    by_type = {}

    print(f"Evaluating {args.results_file}...")
    
    with open(args.results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            data = json.loads(line)
            prob_type = data.get("problem_type", "unknown")
            
            total += 1
            if prob_type not in by_type:
                by_type[prob_type] = {"count": 0, "heuristic_match": 0}
            
            by_type[prob_type]["count"] += 1
            
            # This is a placeholder evaluation.
            # Real evaluation for proofs needs an LLM judge.
            is_match = simple_evaluate(data["solution"], data["generated_solution"])
            
            if is_match:
                correct_heuristic += 1
                by_type[prob_type]["heuristic_match"] += 1

    print(f"Total processed: {total}")
    print("-" * 40)
    print("Breakdown by Problem Type:")
    for p_type, stats in by_type.items():
        acc = (stats["heuristic_match"] / stats["count"]) * 100 if stats["count"] > 0 else 0
        print(f"  {p_type}: {stats['count']} items, {stats['heuristic_match']} heuristic matches ({acc:.2f}%)")
    print("-" * 40)
    print("Note: The heuristic match is very basic (checks if short ground truth is substring of generated output).")
    print("For 'proof' problems, this metric is not reliable. Use an LLM-based judge for accurate evaluation.")

if __name__ == "__main__":
    main()

