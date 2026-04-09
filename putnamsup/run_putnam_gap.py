import os
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from putnam_utils import load_dataset, SUPPORTED_VARIANTS
import json

def run_inference_batch(model, tokenizer, questions: list, device: str) -> list:
    """
    Runs generation for a batch of questions.
    """
    prompts = [f"Problem:\n{q}\n\nPlease solve the problem above step by step and provide the final answer.\n\nSolution:\n" for q in questions]
    
    # Determine target device for inputs
    if device == "auto":
        target_device = model.device
    else:
        target_device = device

    input_texts = []
    if tokenizer.chat_template:
         for p in prompts:
             messages = [{"role": "user", "content": p}]
             try:
                 formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 input_texts.append(formatted)
             except Exception:
                 input_texts.append(p)
    else:
        input_texts = prompts
    
    # Tokenize with padding
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(target_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only new tokens
    # output_ids contains input_ids + new_tokens. We need to slice.
    # However, input lengths might vary due to padding.
    # batch_decode usually decodes everything.
    # A common trick is to decode everything and then strip the prompt, but prompts are different.
    # Better: tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:]) works if left-padded and consistent length?
    # No, with left padding, the new tokens are at the end.
    
    generated_texts = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return [t.strip() for t in generated_texts]

def main():
    parser = argparse.ArgumentParser(description="Run inference on PutnamGAP dataset")
    parser.add_argument("--data_dir", type=str, default="PutnamGAP", help="Path to PutnamGAP JSON files")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--output_file", type=str, default="putnam_gap_results.jsonl", help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of problems to run")
    parser.add_argument("--limit_per_variant", type=int, default=None, help="Limit number of problems per variant")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (use 'auto' for multi-GPU)")
    parser.add_argument("--dry_run", action="store_true", help="Only load data and print first few examples, do not load model")
    parser.add_argument("--variants", type=str, default=None, help=f"Comma-separated list of variants to include. Choices: {','.join(SUPPORTED_VARIANTS)}")
    args = parser.parse_args()

    # Parse variants argument
    selected_variants = None
    
    # Diagnostic check for CUDA availability
    if torch.cuda.device_count() > 0 and not torch.cuda.is_available():
         print("\n" + "!"*60)
         print(f"WARNING: PyTorch detects {torch.cuda.device_count()} CUDA devices but cannot use them.")
         print(f"torch.cuda.is_available() == False")
         print(f"Current PyTorch version: {torch.__version__}")
         print(f"Your driver probably supports an older CUDA version than this PyTorch build.")
         print("!"*60 + "\n")
    
    if args.variants:
        selected_variants = [v.strip() for v in args.variants.split(",")]
        print(f"Filtering for variants: {selected_variants}")

    print(f"Scanning data from {args.data_dir}...")
    dataset = list(load_dataset(args.data_dir, selected_variants=selected_variants))
    print(f"Found {len(dataset)} problem variants.")

    if args.limit_per_variant:
        from collections import defaultdict
        counts = defaultdict(int)
        filtered_dataset = []
        for item in dataset:
            v = item['variant']
            if counts[v] < args.limit_per_variant:
                filtered_dataset.append(item)
                counts[v] += 1
        dataset = filtered_dataset
        print(f"Filtered to {len(dataset)} examples (max {args.limit_per_variant} per variant).")
    
    if args.dry_run:
        if dataset:
            print("\n--- Example 1 ---")
            print(f"Index: {dataset[0]['file_index']}")
            print(f"Variant: {dataset[0]['variant']}")
            print(f"Question: {dataset[0]['question'][:200]}...")
            print(f"Solution: {dataset[0]['solution'][:200]}...")
        return

    print(f"Loading model: {args.model_name_or_path} on {args.device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
        
        # Determine dtype
        torch_dtype = torch.float16
        if args.device == "cpu":
            torch_dtype = torch.float32
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            device_map=args.device, 
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limiting to first {args.limit} examples.")

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        batch_size = args.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), desc="Running Inference"):
            batch = dataset[i : i + batch_size]
            questions = [item["question"] for item in batch]
            
            try:
                generated_answers = run_inference_batch(model, tokenizer, questions, args.device)
            except Exception as e:
                print(f"Error generating for batch starting at index {i}: {e}")
                generated_answers = [f"<ERROR: {str(e)}>" for _ in batch]

            for item, ans in zip(batch, generated_answers):
                result_entry = {
                    "file_index": item["file_index"],
                    "problem_type": item["problem_type"],
                    "variant": item["variant"],
                    "question": item["question"],
                    "solution": item["solution"],
                    "generated_solution": ans
                }
                
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
