#!/usr/bin/env python3
"""
原题 vs Kernel Variant 数学能力对比测试
使用4o-mini解题，o3严格评分，比较两种题目的正确率差异
"""

import os
import json
import asyncio
import pathlib
import time
import re
import random
from typing import Dict, List, Tuple, Optional
import click
import tqdm
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError

# Configuration
SOLVER_MODEL = "gpt-4o-mini"  # 用于解题的模型
GRADER_MODEL = "o3"          # 用于评分的模型
SRC_DIR = pathlib.Path("raw/json")
RESULTS_DIR = pathlib.Path("results/comparison_test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RETRIES = 4
TIMEOUT_BASE = 600
RESP_FMT = {"type": "json_object"}

# 解题系统prompt - 4o-mini
SOLVER_SYSTEM_PROMPT = """You are an expert mathematician solving competition-level problems.
Provide detailed, step-by-step solutions with clear mathematical reasoning.

Requirements:
- Show all your work and intermediate steps
- Justify each major step of your reasoning
- Use proper mathematical notation
- Be thorough but concise
- State your final answer clearly

Solve the problem completely and rigorously."""

SOLVER_USER_TEMPLATE = """Please solve this mathematical problem:

{problem_statement}

Provide a complete solution with detailed reasoning. Return your response in JSON format:
{{"solution": "your complete step-by-step solution with mathematical reasoning",
  "final_answer": "your final answer in a clear, concise form"}}"""

# 证明题严格评分系统prompt - o3
PROOF_GRADER_SYSTEM_PROMPT = """You are an extremely strict mathematical grader evaluating competition-level PROOF problems.

GRADING STANDARDS (BE VERY STRICT):
- Mathematical rigor: Every step must be mathematically sound and justified
- Logical flow: The reasoning must be clear, complete, and logically connected
- Correctness: All calculations, algebraic manipulations, and conclusions must be correct
- Completeness: The solution must address all parts of the problem fully
- Precision: Mathematical statements must be precise and unambiguous

FAILING CRITERIA (Mark as INCORRECT if ANY of these apply):
- Any unjustified logical leap or gap in reasoning
- Any computational error, no matter how small
- Missing steps in critical parts of the argument
- Imprecise or ambiguous mathematical statements
- Incorrect final answer, even if approach is partially correct
- Circular reasoning or logical fallacies
- Misuse of mathematical theorems or definitions

BE EXTREMELY STRICT. Competition mathematics proofs require perfect precision."""

# 计算题相对宽松评分系统prompt - o3  
CALCULATION_GRADER_SYSTEM_PROMPT = """You are a mathematical grader evaluating competition-level CALCULATION problems.

GRADING STANDARDS FOR CALCULATION PROBLEMS:
- Primary focus: Is the final answer correct?
- Secondary focus: Is the overall approach reasonable and mathematically sound?
- Computation: Allow minor computational slips if the method is correct and final answer is right

GRADING CRITERIA:
- CORRECT: Final answer is correct AND approach is fundamentally sound
- INCORRECT: Final answer is wrong OR approach is fundamentally flawed

For calculation problems, the final numerical answer is the most important criterion.
Minor intermediate errors are acceptable if they don't affect the final result."""

PROOF_GRADER_USER_TEMPLATE = """Grade this PROOF solution with extreme strictness.

PROBLEM:
{problem_statement}

STUDENT SOLUTION:
{solution}

CORRECT REFERENCE SOLUTION:
{reference_solution}

Evaluate with maximum strictness. Every logical step must be perfect. Return JSON with:
{{"grade": "CORRECT" or "INCORRECT",
  "detailed_feedback": "specific detailed analysis of what is right/wrong",
  "major_issues": "list of significant mathematical errors or gaps",
  "final_answer_correct": true or false,
  "reasoning_rigor_score": 0-10 integer (10=perfect rigor, 0=severely flawed),
  "overall_assessment": "comprehensive evaluation summary"}}"""

CALCULATION_GRADER_USER_TEMPLATE = """Grade this CALCULATION solution with focus on final answer correctness.

PROBLEM:
{problem_statement}

STUDENT SOLUTION:
{solution}

CORRECT REFERENCE SOLUTION:
{reference_solution}

Focus primarily on whether the final answer is correct. Return JSON with:
{{"grade": "CORRECT" or "INCORRECT",
  "detailed_feedback": "specific detailed analysis of what is right/wrong",
  "major_issues": "list of significant mathematical errors or gaps",
  "final_answer_correct": true or false,
  "reasoning_rigor_score": 0-10 integer (10=perfect rigor, 0=severely flawed),
  "overall_assessment": "comprehensive evaluation summary"}}"""

JSON_RE = re.compile(r"\{[\s\S]*\}")

def parse_json_response(raw: str) -> Optional[Dict]:
    """Parse JSON from LLM response with fallback strategies."""
    if not raw:
        return None
    
    try:
        return json.loads(raw)
    except:
        pass
    
    match = JSON_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    try:
        fixed = raw.replace('\\"', '"').replace('\\\\', '\\')
        return json.loads(fixed)
    except:
        pass
    
    return None

def to_str(x) -> str:
    """Convert various types to string safely."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(map(str, x))
    return str(x)

async def call_api_with_retry(cli: AsyncOpenAI, model: str, messages: List[Dict]) -> Tuple[Optional[Dict], str]:
    """Make OpenAI API call with retry logic."""
    raw_response = ""
    
    for attempt in range(1, RETRIES + 1):
        timeout = TIMEOUT_BASE * (2 ** (attempt - 1))
        try:
            # Set temperature based on model
            # o3, o3-mini, and o4-mini require temperature 1.0
            if any(model_name in model.lower() for model_name in ['o3', 'o3-mini', 'o4-mini']):
                temperature = 1.0
            else:
                # Use temperature 0.0 for deterministic solving with other models
                temperature = 0.0
            
            response = await asyncio.wait_for(
                cli.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format=RESP_FMT,
                ),
                timeout=timeout,
            )
            raw_response = response.choices[0].message.content or ""
            parsed = parse_json_response(raw_response)
            if parsed:
                return parsed, raw_response
            raise ValueError("Failed to parse JSON response")
            
        except RateLimitError as e:
            print(f"🚫 RateLimitError (attempt {attempt}/{RETRIES}): {str(e)}")
            if "insufficient_quota" in str(e):
                print("⏳ Detected quota exhaustion - sleeping 15 minutes")
                await asyncio.sleep(900)
            else:
                sleep_time = 2 ** attempt + random.random()
                print(f"   ⏰ Rate limited, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                
        except (APIError, APIConnectionError, asyncio.TimeoutError, ValueError) as e:
            print(f"❌ {type(e).__name__} (attempt {attempt}/{RETRIES}): {str(e)}")
            if attempt == RETRIES:
                return None, raw_response
            sleep_time = 2 ** attempt + random.random()
            print(f"   ⏰ Retrying in {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
    
    return None, raw_response

async def solve_problem(cli: AsyncOpenAI, problem_statement: str) -> Tuple[Optional[Dict], str]:
    """让4o-mini解题"""
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": SOLVER_USER_TEMPLATE.format(
            problem_statement=problem_statement
        )}
    ]
    return await call_api_with_retry(cli, SOLVER_MODEL, messages)

async def grade_solution(cli: AsyncOpenAI, problem_statement: str, solution: str, 
                        reference_solution: str, problem_type: str = "proof") -> Tuple[Optional[Dict], str]:
    """让o3根据题型评分 - 证明题严格，计算题注重答案"""
    if problem_type == "calculation":
        system_prompt = CALCULATION_GRADER_SYSTEM_PROMPT
        user_template = CALCULATION_GRADER_USER_TEMPLATE
    else:  # Default to proof (strict grading)
        system_prompt = PROOF_GRADER_SYSTEM_PROMPT
        user_template = PROOF_GRADER_USER_TEMPLATE
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(
            problem_statement=problem_statement,
            solution=solution,
            reference_solution=reference_solution
        )}
    ]
    return await call_api_with_retry(cli, GRADER_MODEL, messages)

async def test_single_file(file_path: pathlib.Path, cli: AsyncOpenAI) -> Dict:
    """测试单个文件的原题和kernel variant"""
    try:
        # 加载数据
        data = json.loads(file_path.read_text(encoding='utf-8'))
        index = data.get("index", file_path.stem)
        
        # 检查必要字段
        original_question = to_str(data.get("question", "")).strip()
        original_solution = to_str(data.get("solution", "")).strip()
        problem_type = data.get("problem_type", "proof")  # 默认为证明题，严格评分
        
        kv = data.get("variants", {}).get("kernel_variant")
        if not kv:
            return {
                "index": index,
                "status": "skipped",
                "reason": "no_kernel_variant"
            }
        
        kernel_question = to_str(kv.get("question", "")).strip()
        kernel_solution = to_str(kv.get("solution", "")).strip()
        
        if not all([original_question, original_solution, kernel_question, kernel_solution]):
            return {
                "index": index,
                "status": "skipped", 
                "reason": "missing_fields"
            }
        
        print(f"🧮 Testing {index} (Type: {problem_type.upper()})")
        start_time = time.time()
        
        result = {
            "index": index,
            "status": "completed",
            "timestamp": time.time(),
            "problem_type": problem_type,
            "original": {},
            "kernel_variant": {},
            "comparison": {}
        }
        
        # 1. 让4o-mini解原题
        print(f"   📝 Solving original problem...")
        orig_solve_result, orig_solve_raw = await solve_problem(cli, original_question)
        
        if not orig_solve_result:
            result["original"]["solve_status"] = "failed"
            result["status"] = "failed"
            return result
        
        orig_student_solution = to_str(orig_solve_result.get("solution", "")).strip()
        orig_final_answer = to_str(orig_solve_result.get("final_answer", "")).strip()
        
        result["original"]["student_solution"] = orig_student_solution
        result["original"]["student_final_answer"] = orig_final_answer
        result["original"]["solve_status"] = "success"
        
        # 2. 让4o-mini解kernel variant
        print(f"   📝 Solving kernel variant...")
        kv_solve_result, kv_solve_raw = await solve_problem(cli, kernel_question)
        
        if not kv_solve_result:
            result["kernel_variant"]["solve_status"] = "failed"
            result["status"] = "failed"
            return result
        
        kv_student_solution = to_str(kv_solve_result.get("solution", "")).strip()
        kv_final_answer = to_str(kv_solve_result.get("final_answer", "")).strip()
        
        result["kernel_variant"]["student_solution"] = kv_student_solution
        result["kernel_variant"]["student_final_answer"] = kv_final_answer
        result["kernel_variant"]["solve_status"] = "success"
        
        # 3. o3根据题型评分原题解答
        grading_style = "STRICT" if problem_type == "proof" else "LENIENT"
        print(f"   🔍 Grading original solution ({grading_style})...")
        orig_grade_result, orig_grade_raw = await grade_solution(
            cli, original_question, orig_student_solution, original_solution, problem_type
        )
        
        if not orig_grade_result:
            result["original"]["grade_status"] = "failed"
        else:
            result["original"]["grade_status"] = "success"
            result["original"]["grade"] = orig_grade_result.get("grade", "UNKNOWN")
            result["original"]["detailed_feedback"] = orig_grade_result.get("detailed_feedback", "")
            result["original"]["major_issues"] = orig_grade_result.get("major_issues", "")
            result["original"]["final_answer_correct"] = orig_grade_result.get("final_answer_correct", False)
            result["original"]["reasoning_rigor_score"] = orig_grade_result.get("reasoning_rigor_score", 0)
            result["original"]["overall_assessment"] = orig_grade_result.get("overall_assessment", "")
        
        # 4. o3根据题型评分kernel variant解答
        print(f"   🔍 Grading kernel variant solution ({grading_style})...")
        kv_grade_result, kv_grade_raw = await grade_solution(
            cli, kernel_question, kv_student_solution, kernel_solution, problem_type
        )
        
        if not kv_grade_result:
            result["kernel_variant"]["grade_status"] = "failed"
        else:
            result["kernel_variant"]["grade_status"] = "success"
            result["kernel_variant"]["grade"] = kv_grade_result.get("grade", "UNKNOWN")
            result["kernel_variant"]["detailed_feedback"] = kv_grade_result.get("detailed_feedback", "")
            result["kernel_variant"]["major_issues"] = kv_grade_result.get("major_issues", "")
            result["kernel_variant"]["final_answer_correct"] = kv_grade_result.get("final_answer_correct", False)
            result["kernel_variant"]["reasoning_rigor_score"] = kv_grade_result.get("reasoning_rigor_score", 0)
            result["kernel_variant"]["overall_assessment"] = kv_grade_result.get("overall_assessment", "")
        
        # 5. 比较分析
        if (result["original"]["grade_status"] == "success" and 
            result["kernel_variant"]["grade_status"] == "success"):
            
            orig_correct = result["original"]["grade"] == "CORRECT"
            kv_correct = result["kernel_variant"]["grade"] == "CORRECT"
            
            result["comparison"]["original_correct"] = orig_correct
            result["comparison"]["kernel_variant_correct"] = kv_correct
            result["comparison"]["both_correct"] = orig_correct and kv_correct
            result["comparison"]["both_incorrect"] = not orig_correct and not kv_correct
            result["comparison"]["original_harder"] = not orig_correct and kv_correct  # 原题更难
            result["comparison"]["kernel_variant_harder"] = orig_correct and not kv_correct  # kernel variant更难
            
            orig_rigor = result["original"]["reasoning_rigor_score"]
            kv_rigor = result["kernel_variant"]["reasoning_rigor_score"]
            result["comparison"]["rigor_difference"] = orig_rigor - kv_rigor  # 正数=原题推理更严谨
        
        total_time = time.time() - start_time
        result["processing_time"] = total_time
        
        print(f"   ✅ Completed {index} in {total_time:.1f}s")
        if result["comparison"]:
            orig_status = "✅" if result["comparison"]["original_correct"] else "❌"
            kv_status = "✅" if result["comparison"]["kernel_variant_correct"] else "❌"
            print(f"      Original: {orig_status}, Kernel Variant: {kv_status}")
        
        return result
        
    except Exception as e:
        return {
            "index": index if 'index' in locals() else file_path.stem,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }

async def save_detailed_results(results: List[Dict], output_file: str):
    """保存详细结果"""
    output_path = RESULTS_DIR / f"{output_file}_detailed.json"
    try:
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"💾 Detailed results saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save detailed results: {e}")

def generate_summary_report(results: List[Dict]) -> Dict:
    """生成汇总报告"""
    summary = {
        "total_files": len(results),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "by_problem_type": {
            "proof": {"count": 0, "original_correct": 0, "kv_correct": 0},
            "calculation": {"count": 0, "original_correct": 0, "kv_correct": 0}
        },
        "original_stats": {"correct": 0, "incorrect": 0, "total_graded": 0},
        "kernel_variant_stats": {"correct": 0, "incorrect": 0, "total_graded": 0},
        "comparison_stats": {
            "both_correct": 0,
            "both_incorrect": 0,
            "original_harder": 0,
            "kernel_variant_harder": 0,
            "total_compared": 0
        },
        "rigor_analysis": {
            "original_avg_rigor": 0,  
            "kernel_variant_avg_rigor": 0,
            "rigor_difference_avg": 0
        }
    }
    
    orig_rigor_scores = []
    kv_rigor_scores = []
    rigor_differences = []
    
    for result in results:
        if result["status"] == "completed":
            summary["completed"] += 1
            
            # 按题型统计
            ptype = result.get("problem_type", "proof")
            if ptype in summary["by_problem_type"]:
                summary["by_problem_type"][ptype]["count"] += 1
                if result["original"].get("grade") == "CORRECT":
                    summary["by_problem_type"][ptype]["original_correct"] += 1
                if result["kernel_variant"].get("grade") == "CORRECT":
                    summary["by_problem_type"][ptype]["kv_correct"] += 1
            
            # 原题统计
            if result["original"].get("grade_status") == "success":
                summary["original_stats"]["total_graded"] += 1
                if result["original"]["grade"] == "CORRECT":
                    summary["original_stats"]["correct"] += 1
                else:
                    summary["original_stats"]["incorrect"] += 1
                orig_rigor_scores.append(result["original"]["reasoning_rigor_score"])
            
            # kernel variant统计
            if result["kernel_variant"].get("grade_status") == "success":
                summary["kernel_variant_stats"]["total_graded"] += 1
                if result["kernel_variant"]["grade"] == "CORRECT":
                    summary["kernel_variant_stats"]["correct"] += 1
                else:
                    summary["kernel_variant_stats"]["incorrect"] += 1
                kv_rigor_scores.append(result["kernel_variant"]["reasoning_rigor_score"])
            
            # 比较统计
            if result.get("comparison"):
                summary["comparison_stats"]["total_compared"] += 1
                comp = result["comparison"]
                if comp["both_correct"]:
                    summary["comparison_stats"]["both_correct"] += 1
                elif comp["both_incorrect"]:
                    summary["comparison_stats"]["both_incorrect"] += 1
                elif comp["original_harder"]:
                    summary["comparison_stats"]["original_harder"] += 1
                elif comp["kernel_variant_harder"]:
                    summary["comparison_stats"]["kernel_variant_harder"] += 1
                
                rigor_differences.append(comp["rigor_difference"])
        
        elif result["status"] == "skipped":
            summary["skipped"] += 1
        else:
            summary["failed"] += 1
    
    # 计算平均分
    if orig_rigor_scores:
        summary["rigor_analysis"]["original_avg_rigor"] = sum(orig_rigor_scores) / len(orig_rigor_scores)
    if kv_rigor_scores:
        summary["rigor_analysis"]["kernel_variant_avg_rigor"] = sum(kv_rigor_scores) / len(kv_rigor_scores)
    if rigor_differences:
        summary["rigor_analysis"]["rigor_difference_avg"] = sum(rigor_differences) / len(rigor_differences)
    
    # 计算正确率
    if summary["original_stats"]["total_graded"] > 0:
        summary["original_stats"]["accuracy"] = summary["original_stats"]["correct"] / summary["original_stats"]["total_graded"]
    
    if summary["kernel_variant_stats"]["total_graded"] > 0:
        summary["kernel_variant_stats"]["accuracy"] = summary["kernel_variant_stats"]["correct"] / summary["kernel_variant_stats"]["total_graded"]
    
    return summary

def print_summary_report(summary: Dict):
    """打印汇总报告"""
    print("\n" + "="*80)
    print("📊 ORIGINAL vs KERNEL VARIANT COMPARISON REPORT")
    print("="*80)
    
    print(f"📁 Total files: {summary['total_files']}")
    print(f"✅ Completed: {summary['completed']}")
    print(f"⏭️ Skipped: {summary['skipped']}")
    print(f"❌ Failed: {summary['failed']}")
    
    print(f"\n📈 ACCURACY COMPARISON:")
    orig_acc = summary["original_stats"].get("accuracy", 0) * 100
    kv_acc = summary["kernel_variant_stats"].get("accuracy", 0) * 100
    print(f"Original Problems:     {orig_acc:.1f}% ({summary['original_stats']['correct']}/{summary['original_stats']['total_graded']})")
    print(f"Kernel Variants:       {kv_acc:.1f}% ({summary['kernel_variant_stats']['correct']}/{summary['kernel_variant_stats']['total_graded']})")
    
    if orig_acc > 0 and kv_acc > 0:
        diff = orig_acc - kv_acc
        if diff > 5:
            print(f"📉 Kernel variants are {diff:.1f}% harder (as expected)")
        elif diff < -5:
            print(f"📈 Original problems are {-diff:.1f}% harder (unexpected)")
        else:
            print(f"📊 Similar difficulty (difference: {diff:.1f}%)")
    
    print(f"\n🎯 BY PROBLEM TYPE:")
    for ptype, stats in summary["by_problem_type"].items():
        if stats["count"] > 0:
            orig_acc_type = (stats["original_correct"] / stats["count"]) * 100
            kv_acc_type = (stats["kv_correct"] / stats["count"]) * 100
            grading_note = " (STRICT grading)" if ptype == "proof" else " (LENIENT grading)"
            print(f"{ptype.upper()} Problems{grading_note}:")
            print(f"  Original:      {orig_acc_type:.1f}% ({stats['original_correct']}/{stats['count']})")
            print(f"  Kernel Variant: {kv_acc_type:.1f}% ({stats['kv_correct']}/{stats['count']})")
            if stats["count"] >= 3:  # Only show difference if we have enough samples
                type_diff = orig_acc_type - kv_acc_type
                print(f"  Difference:    {type_diff:+.1f}%")
    
    print(f"\n🔍 DETAILED COMPARISON:")
    comp = summary["comparison_stats"]
    total = comp["total_compared"]
    if total > 0:
        print(f"Both correct:          {comp['both_correct']:3d} ({comp['both_correct']/total*100:.1f}%)")
        print(f"Both incorrect:        {comp['both_incorrect']:3d} ({comp['both_incorrect']/total*100:.1f}%)")
        print(f"Original harder:       {comp['original_harder']:3d} ({comp['original_harder']/total*100:.1f}%)")
        print(f"Kernel variant harder: {comp['kernel_variant_harder']:3d} ({comp['kernel_variant_harder']/total*100:.1f}%)")
    
    print(f"\n📏 REASONING RIGOR ANALYSIS:")
    rigor = summary["rigor_analysis"]
    print(f"Original avg rigor:    {rigor['original_avg_rigor']:.2f}/10")
    print(f"Kernel variant rigor:  {rigor['kernel_variant_avg_rigor']:.2f}/10")
    print(f"Difference:            {rigor['rigor_difference_avg']:.2f} (positive = original more rigorous)")
    
    print("="*80)

@click.command()
@click.option("-c", "--concurrency", default=16, show_default=True,
              help="Maximum concurrent processing tasks")
@click.option("--max-files", default=50, show_default=True,
              help="Maximum number of files to test (for quick testing)")
@click.option("--file-pattern", default="*.json", show_default=True,
              help="File pattern to process")
@click.option("--output-prefix", default="comparison_test", show_default=True,
              help="Prefix for output files")
@click.option("--debug", is_flag=True, help="Enable debug output")
def main(concurrency: int, max_files: int, file_pattern: str, output_prefix: str, debug: bool):
    """原题 vs Kernel Variant 数学能力对比测试"""
    print(f"🧪 Starting Original vs Kernel Variant Comparison Test")
    print(f"   Solver Model: {SOLVER_MODEL}")
    print(f"   Grader Model: {GRADER_MODEL}")
    print(f"   Max files: {max_files}")
    print(f"   Concurrency: {concurrency}")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        return
    
    # 找到测试文件
    all_files = sorted(SRC_DIR.glob(file_pattern))
    if max_files > 0:
        all_files = all_files[:max_files]
    
    print(f"📁 Testing {len(all_files)} files")
    
    if not all_files:
        print("❌ No files found to test!")
        return
    
    async def run_test():
        cli = AsyncOpenAI()
        sem = asyncio.Semaphore(concurrency)
        
        async def worker(file_path: pathlib.Path):
            async with sem:
                return await test_single_file(file_path, cli)
        
        # 执行测试
        results = []
        progress_bar = tqdm.tqdm(total=len(all_files), desc="Testing", unit="file")
        
        tasks = [worker(f) for f in all_files]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress_bar.update(1)
        
        progress_bar.close()
        return results
    
    # 运行测试
    results = asyncio.run(run_test())
    
    # 保存详细结果
    timestamp = int(time.time())
    output_name = f"{output_prefix}_{timestamp}"
    asyncio.run(save_detailed_results(results, output_name))
    
    # 生成并显示汇总报告
    summary = generate_summary_report(results)
    print_summary_report(summary)
    
    # 保存汇总报告
    summary_path = RESULTS_DIR / f"{output_name}_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"💾 Summary report saved to {summary_path}")
    except Exception as e:
        print(f"❌ Failed to save summary: {e}")

if __name__ == "__main__":
    main()
 