import os
import json
import argparse
import asyncio
import time
from tqdm.asyncio import tqdm
from putnam_utils import load_dataset, SUPPORTED_VARIANTS

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

async def process_item(sem, client, model_name, item):
    """
    Process a single item with semaphore for concurrency control.
    """
    async with sem:
        question = item["question"]
        prompt = f"Problem:\n{question}\n\nPlease solve the problem above step by step and provide the final answer.\n\nSolution:\n"
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Call API asynchronously
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                extra_headers={
                   "HTTP-Referer": "https://github.com/PutnamGAP",
                   "X-Title": "PutnamGAP Eval",
                }
            )
            generated_answer = completion.choices[0].message.content
        except Exception as e:
            generated_answer = f"<API ERROR: {str(e)}>"

        # Construct result entry
        result_entry = {
            "file_index": item["file_index"],
            "problem_type": item["problem_type"],
            "variant": item["variant"],
            "question": question,
            "solution": item["solution"],
            "generated_solution": generated_answer,
            "model": model_name
        }
        return result_entry

async def run_async_inference(args, dataset):
    if AsyncOpenAI is None:
        print("Error: 'openai' library not found. Please install it via: pip install openai")
        return

    if not args.api_key:
        print("Error: API key not provided. Use --api_key or set OPENROUTER_API_KEY env var.")
        return

    print(f"Initializing AsyncOpenAI client with base_url={args.base_url}")
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    concurrency = args.concurrency
    print(f"Running with concurrency: {concurrency}")
    sem = asyncio.Semaphore(concurrency)
    
    tasks = []
    for item in dataset:
        task = process_item(sem, client, args.model_name, item)
        tasks.append(task)

    print(f"Starting {len(tasks)} tasks using model: {args.model_name}")
    
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Inference"):
            result = await future
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Done. Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on PutnamGAP dataset via OpenRouter (Async)")
    parser.add_argument("--data_dir", type=str, default="PutnamGAP", help="Path to PutnamGAP JSON files")
    parser.add_argument("--model_name", type=str, required=True, help="OpenRouter model name")
    parser.add_argument("--output_file", type=str, default="putnam_gap_openrouter_results.jsonl", help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to run (for testing)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API Key")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1", help="API Base URL")
    parser.add_argument("--dry_run", action="store_true", help="Only load data and print info")
    parser.add_argument("--variants", type=str, default=None, help=f"Comma-separated list of variants to include. Choices: {','.join(SUPPORTED_VARIANTS)}")
    
    args = parser.parse_args()

    # Parse variants argument
    selected_variants = None
    if args.variants:
        selected_variants = [v.strip() for v in args.variants.split(",")]
        print(f"Filtering for variants: {selected_variants}")
    
    print(f"Scanning data from {args.data_dir}...")
    dataset = list(load_dataset(args.data_dir, selected_variants=selected_variants))
    print(f"Found {len(dataset)} problem variants.")

    if args.dry_run:
        if dataset:
            print("\n--- Example 1 ---")
            print(f"Index: {dataset[0]['file_index']}")
            print(f"Variant: {dataset[0]['variant']}")
            print(f"Question: {dataset[0]['question'][:200]}...")
            return

    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limiting to first {args.limit} examples.")

    asyncio.run(run_async_inference(args, dataset))

if __name__ == "__main__":
    main()
