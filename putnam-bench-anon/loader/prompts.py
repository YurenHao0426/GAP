"""
Prompt templates for mathematical problem solving and grading.
These prompts have been refined and validated through extensive testing.
"""

# Solver system prompt - 4o-mini
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

# Proof strict grading system prompt - o3
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

# Calculation lenient grading system prompt - o3  
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

# Response format for JSON output
RESPONSE_FORMAT = {"type": "json_object"}

# Default retry and timeout settings
DEFAULT_RETRIES = 6  # Limited to 6 retries before marking as failed
DEFAULT_TIMEOUT_BASE = 600 