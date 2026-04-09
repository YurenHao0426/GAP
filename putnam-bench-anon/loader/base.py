"""
Abstract base class for model loaders.
Defines the interface for mathematical problem solving and grading.
"""

import re
import json
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

from .prompts import (
    SOLVER_SYSTEM_PROMPT,
    SOLVER_USER_TEMPLATE,
    PROOF_GRADER_SYSTEM_PROMPT,
    CALCULATION_GRADER_SYSTEM_PROMPT,
    PROOF_GRADER_USER_TEMPLATE,
    CALCULATION_GRADER_USER_TEMPLATE,
    RESPONSE_FORMAT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_BASE
)

# JSON extraction regex
JSON_RE = re.compile(r"\{[\s\S]*\}")


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    def __init__(self, 
                 solver_model: str,
                 grader_model: str,
                 retries: int = DEFAULT_RETRIES,
                 timeout_base: int = DEFAULT_TIMEOUT_BASE,
                 debug: bool = False,
                 quick: bool = False):
        """
        Initialize the model loader.
        
        Args:
            solver_model: Model name for solving problems
            grader_model: Model name for grading solutions
            retries: Number of retry attempts for API calls
            timeout_base: Base timeout in seconds for API calls
            debug: Enable debug logging for JSON parsing
            quick: Quick mode - allows one retry with 1200s timeout each attempt
        """
        self.solver_model = solver_model
        self.grader_model = grader_model
        self.retries = retries
        self.timeout_base = timeout_base
        self.debug = debug
        self.quick = quick
        
        # Override settings for quick mode
        if self.quick:
            self.retries = 1  # Only try once
            self.timeout_base = 1200  # 20 minutes timeout
    
    @abstractmethod
    async def _call_api(self, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Make an API call to the model.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (parsed_response, raw_response)
        """
        pass
    
    def parse_json_response(self, raw: str, debug: bool = False) -> Optional[Dict]:
        """Parse JSON from LLM response with fallback strategies."""
        if not raw:
            return None
        
        # Try direct JSON parse
        try:
            return json.loads(raw)
        except Exception as e:
            if debug:
                print(f"⚠️ Direct JSON parse failed: {e}")
        
        # Try to find JSON in the response
        match = JSON_RE.search(raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                if debug:
                    print(f"⚠️ Regex JSON parse failed: {e}")
        
        # Try fixing common JSON issues including control characters
        try:
            # Fix escaped quotes and backslashes
            fixed = raw.replace('\\"', '"').replace('\\\\', '\\')
            
            # Fix unescaped newlines and other control characters in JSON strings
            # This is a more robust approach for LLM responses
            import ast
            
            # Try to use ast.literal_eval if it's a simple dict-like structure
            if fixed.strip().startswith('{') and fixed.strip().endswith('}'):
                try:
                    # Replace common problematic patterns
                    fixed = fixed.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    return json.loads(fixed)
                except Exception as e:
                    if debug:
                        print(f"⚠️ Fixed JSON parse failed: {e}")
            
        except Exception as e:
            if debug:
                print(f"⚠️ JSON fixing failed: {e}")
        
        # ENHANCED: Try to complete truncated JSON
        try:
            if raw.strip().startswith('{') and not raw.strip().endswith('}'):
                if debug:
                    print("🔧 Attempting to fix truncated JSON...")
                
                # Try to find the last complete key-value pair
                # Look for solution content
                if '"solution"' in raw:
                    # Extract solution up to the truncation point
                    solution_start = raw.find('"solution"')
                    solution_content = raw[solution_start:]
                    
                    # Find the actual solution text
                    import re
                    solution_match = re.search(r'"solution":\s*"([^"]*(?:\\"[^"]*)*)', raw, re.DOTALL)
                    if solution_match:
                        solution_text = solution_match.group(1)
                        # Clean up the solution text
                        solution_text = solution_text.replace('\\"', '"').replace('\\n', '\n')
                        
                        if debug:
                            print(f"🔧 Extracted solution from truncated JSON ({len(solution_text)} chars)")
                        return {
                            "solution": solution_text,
                            "final_answer": "Solution was truncated - see solution field for complete answer"
                        }
        except Exception as e:
            if debug:
                print(f"⚠️ Truncated JSON recovery failed: {e}")
        
        # Final fallback: try to extract key-value pairs manually
        try:
            if '"solution"' in raw:
                import re
                
                if debug:
                    print("🔧 Attempting manual key-value extraction...")
                
                # Extract solution (more robust pattern)
                solution_match = re.search(r'"solution":\s*"([^"]*(?:\\"[^"]*)*)', raw, re.DOTALL)
                solution = solution_match.group(1) if solution_match else ""
                
                # Extract final_answer if it exists
                answer_match = re.search(r'"final_answer":\s*"([^"]*)"', raw)
                final_answer = answer_match.group(1) if answer_match else ""
                
                if solution:
                    # Clean up the solution text
                    solution = solution.replace('\\"', '"').replace('\\n', '\n')
                    
                    if debug:
                        print(f"🔧 Manual extraction successful ({len(solution)} chars solution)")
                    return {
                        "solution": solution,
                        "final_answer": final_answer if final_answer else "See solution field for complete answer"
                    }
        except Exception as e:
            if debug:
                print(f"⚠️ Manual extraction failed: {e}")
        
        if debug:
            print("❌ All JSON parsing strategies failed")
        return None
    
    def to_str(self, x) -> str:
        """Convert various types to string safely."""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            return "\n".join(map(str, x))
        return str(x)
    
    async def call_api_with_retry(self, 
                                 model: str, 
                                 messages: List[Dict[str, str]], 
                                 temperature: float = 0.0) -> Tuple[Optional[Dict], str]:
        """
        Make API call with retry logic and JSON parsing.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (parsed_json_response, raw_response)
        """
        raw_response = ""
        
        # In quick mode, we allow one retry with a fixed timeout
        if self.quick:
            max_attempts = 2  # Allow one retry in quick mode
            if self.debug:
                print(f"⚡ Quick mode: Up to {max_attempts} attempts with {self.timeout_base}s timeout each")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1 and self.debug:
                        print(f"🔄 Quick mode retry attempt {attempt}/{max_attempts}")
                    
                    parsed, raw_response = await asyncio.wait_for(
                        self._call_api(model, messages, temperature),
                        timeout=self.timeout_base
                    )
                    
                    if parsed:
                        # Try to parse as JSON
                        debug_mode = getattr(self, 'debug', False)
                        json_parsed = self.parse_json_response(parsed, debug=debug_mode)
                        if json_parsed:
                            return json_parsed, raw_response
                        return None, raw_response
                    else:
                        raise ValueError("Empty response from API")
                        
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    print(f"❌ {error_type} in quick mode (attempt {attempt}/{max_attempts}): {error_msg}")
                    
                    # If this was the last attempt, mark as failed
                    if attempt == max_attempts:
                        return {"_max_retries_reached": True, "error": str(e)}, raw_response
                    
                    # Otherwise, wait a bit before retrying
                    if self.debug:
                        print("⏳ Waiting 5 seconds before retry...")
                    await asyncio.sleep(5)
        
        # Regular mode with retries
        for attempt in range(1, self.retries + 1):
            # More aggressive timeout scaling for persistent failures
            # Cap timeout at 30 minutes to prevent extremely long waits
            timeout = min(self.timeout_base * (1.5 ** (attempt - 1)), 1800)
            if self.debug:
                print(f"🔄 Attempt {attempt}/{self.retries} with timeout {timeout:.0f}s")
            try:
                parsed, raw_response = await asyncio.wait_for(
                    self._call_api(model, messages, temperature),
                    timeout=timeout
                )
                
                if parsed:
                    # Try to parse as JSON
                    debug_mode = getattr(self, 'debug', False)
                    json_parsed = self.parse_json_response(parsed, debug=debug_mode)
                    if json_parsed:
                        return json_parsed, raw_response
                    return None, raw_response
                else:
                    raise ValueError("Empty response from API")
                    
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Only show detailed error info on first attempt or in debug mode
                if attempt == 1 or self.debug:
                    print(f"❌ {error_type} (attempt {attempt}/{self.retries}): {error_msg}")
                
                if attempt == self.retries:
                    print(f"🔥 All {self.retries} retry attempts exhausted for {error_type}")
                    # Return a special marker for max retries reached
                    return {"_max_retries_reached": True, "error": str(e)}, raw_response
                
                # Custom retry strategy: 600s -> 900s -> 900s -> 1200s...
                if attempt == 1:
                    base_sleep = 600  # 10 minutes
                elif attempt == 2 or attempt == 3:
                    base_sleep = 900  # 15 minutes
                else:
                    base_sleep = 1200  # 20 minutes
                
                # Add small random jitter to avoid synchronized retries
                jitter = random.uniform(0, 30)  # 0-30 seconds jitter
                sleep_time = base_sleep + jitter
                
                if self.debug:
                    print(f"   ⏰ Using custom backoff strategy: {base_sleep}s base + {jitter:.1f}s jitter")
                
                if self.debug:
                    print(f"   ⏰ Retrying in {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        return None, raw_response
    
    async def solve_problem(self, problem_statement: str, model: Optional[str] = None) -> Tuple[Optional[Dict], str]:
        """
        Have model solve mathematical problems.
        
        Args:
            problem_statement: Problem statement
            model: Model name to use for solving (if None, uses default solver_model)
            
        Returns:
            Tuple of (solving result dictionary, raw response)
            Solving result contains: {"solution": "detailed solution", "final_answer": "final answer"}
        """
        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": SOLVER_USER_TEMPLATE.format(
                problem_statement=problem_statement
            )}
        ]
        
        # Use specified model or default solver model
        solver_model = model if model is not None else self.solver_model
        
        # Set temperature based on model
        # o3, o3-mini, and o4-mini require temperature 1.0
        if any(model_name in solver_model.lower() for model_name in ['o3', 'o3-mini', 'o4-mini']):
            temperature = 1.0
        else:
            # Use temperature 0.0 for deterministic solving with other models
            temperature = 0.0
        
        return await self.call_api_with_retry(solver_model, messages, temperature=temperature)
    
    async def grade_solution(self, 
                           problem_statement: str, 
                           solution: str,
                           reference_solution: str, 
                           problem_type: str = "proof",
                           model: Optional[str] = None) -> Tuple[Optional[Dict], str]:
        """
        Have model grade solution based on problem type.
        
        Args:
            problem_statement: Problem statement
            solution: Student solution
            reference_solution: Reference solution
            problem_type: Problem type ("proof" strict grading, "calculation" lenient grading)
            model: Model name to use for grading (if None, uses default grader_model)
            
        Returns:
            Tuple of (grading result dictionary, raw response)
            Grading result contains: {"grade": "CORRECT"/"INCORRECT", "detailed_feedback": "...", ...}
        """
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
        
        # Use specified model or default grader model
        grader_model = model if model is not None else self.grader_model
        
        # Use temperature 1.0 for grading (as per original script for o3)
        return await self.call_api_with_retry(grader_model, messages, temperature=1.0)
    
    async def test_single_problem(self, 
                                data: Dict, 
                                variant_type: str = "original",
                                solver_model: Optional[str] = None,
                                grader_model: Optional[str] = None) -> Dict:
        """
        Test complete workflow for single problem: solving + grading.
        
        Args:
            data: Problem data dictionary
            variant_type: Problem variant type ("original" or key names in variants)
            solver_model: Model name for solving (if None, uses default solver_model)
            grader_model: Model name for grading (if None, uses default grader_model)
            
        Returns:
            Test result dictionary
        """
        index = data.get("index", "unknown")
        problem_type = data.get("problem_type", "proof")
        
        try:
            # Get problem and reference solution
            if variant_type == "original":
                question = self.to_str(data.get("question", "")).strip()
                reference_solution = self.to_str(data.get("solution", "")).strip()
            else:
                variant = data.get("variants", {}).get(variant_type)
                if not variant:
                    return {
                        "index": index,
                        "variant_type": variant_type,
                        "status": "skipped",
                        "reason": f"no_{variant_type}_variant"
                    }
                question = self.to_str(variant.get("question", "")).strip()
                reference_solution = self.to_str(variant.get("solution", "")).strip()
            
            if not question or not reference_solution:
                return {
                    "index": index,
                    "variant_type": variant_type,
                    "status": "skipped",
                    "reason": "missing_fields"
                }
            
            result = {
                "index": index,
                "variant_type": variant_type,
                "problem_type": problem_type,
                "status": "completed",
                "solve": {},
                "grade": {}
            }
            
            # 1. Solve problem
            solve_result, solve_raw = await self.solve_problem(question, model=solver_model)
            
            # Check if max retries reached
            if solve_result and solve_result.get("_max_retries_reached"):
                # Mark as completed but with INCORRECT grade due to max retries
                result["solve"]["status"] = "max_retries"
                result["solve"]["solution"] = "Failed to generate solution after maximum retry attempts"
                result["solve"]["final_answer"] = "No answer - max retries reached"
                result["grade"]["status"] = "auto_failed"
                result["grade"]["grade"] = "INCORRECT"
                result["grade"]["detailed_feedback"] = f"Automatically marked as incorrect due to reaching maximum retry limit ({self.retries} attempts)"
                result["grade"]["major_issues"] = "API call failed after all retry attempts"
                result["grade"]["final_answer_correct"] = False
                result["grade"]["reasoning_rigor_score"] = 0
                result["grade"]["overall_assessment"] = "Failed to generate solution"
                result["correct"] = False
                result["status"] = "completed"  # Mark as completed, not failed
                return result
            
            if not solve_result:
                result["solve"]["status"] = "failed"
                result["status"] = "failed"
                return result
            
            student_solution = self.to_str(solve_result.get("solution", "")).strip()
            final_answer = self.to_str(solve_result.get("final_answer", "")).strip()
            
            result["solve"]["status"] = "success"
            result["solve"]["solution"] = student_solution
            result["solve"]["final_answer"] = final_answer
            
            # 2. Grade solution
            grade_result, grade_raw = await self.grade_solution(
                question, student_solution, reference_solution, problem_type, model=grader_model
            )
            
            # Check if grading max retries reached
            if grade_result and grade_result.get("_max_retries_reached"):
                # Mark as completed but with INCORRECT grade due to max retries in grading
                result["grade"]["status"] = "auto_failed"
                result["grade"]["grade"] = "INCORRECT"
                result["grade"]["detailed_feedback"] = f"Automatically marked as incorrect due to grading reaching maximum retry limit ({self.retries} attempts)"
                result["grade"]["major_issues"] = "Grading API call failed after all retry attempts"
                result["grade"]["final_answer_correct"] = False
                result["grade"]["reasoning_rigor_score"] = 0
                result["grade"]["overall_assessment"] = "Failed to grade solution"
                result["correct"] = False
                result["status"] = "completed"  # Mark as completed, not partial/failed
            elif not grade_result:
                result["grade"]["status"] = "failed"
                result["status"] = "partial"  # solving succeeded but grading failed
            else:
                result["grade"]["status"] = "success"
                result["grade"]["grade"] = grade_result.get("grade", "UNKNOWN")
                result["grade"]["detailed_feedback"] = grade_result.get("detailed_feedback", "")
                result["grade"]["major_issues"] = grade_result.get("major_issues", "")
                result["grade"]["final_answer_correct"] = grade_result.get("final_answer_correct", False)
                result["grade"]["reasoning_rigor_score"] = grade_result.get("reasoning_rigor_score", 0)
                result["grade"]["overall_assessment"] = grade_result.get("overall_assessment", "")
                
                # Mark whether correct
                result["correct"] = grade_result.get("grade") == "CORRECT"
            
            return result
            
        except Exception as e:
            return {
                "index": index,
                "variant_type": variant_type,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
