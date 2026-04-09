"""Async API caller for rescue experiment.

Supports OpenAI, Anthropic, Google. All callers return a unified dict:
  {"status": "success"|"failed", "content": str, "error": str|None}

Concurrency is controlled per-provider via asyncio.Semaphore so we don't
saturate rate limits in any one provider.
"""
from __future__ import annotations
import asyncio
import json
import os
import random
from typing import Optional

# ---------- Provider constants ----------

# Solver model -> provider mapping
SOLVER_PROVIDERS = {
    "gpt-4.1-mini":      "openai",
    "gpt-4o-mini":       "openai",
    "claude-sonnet-4":   "anthropic",
    "gemini-2.5-flash":  "google",
}

# API model strings (the canonical IDs to send)
API_MODEL_NAMES = {
    "gpt-4.1-mini":     "gpt-4.1-mini",
    "gpt-4o-mini":      "gpt-4o-mini",
    "claude-sonnet-4":  "claude-sonnet-4-20250514",
    "gemini-2.5-flash": "gemini-2.5-flash",
}

GRADER_MODEL = "gpt-4o"
GRADER_PROVIDER = "openai"

PER_PROVIDER_CONCURRENCY = {
    "openai": 500,
    "anthropic": 25,   # 90k tok/min cap; 25 in flight keeps us comfortably under
    "google": 300,
}

DEFAULT_RETRIES = 6
DEFAULT_BASE_TIMEOUT = 300.0
RATE_LIMIT_BACKOFF_SECONDS = 60.0  # min sleep on rate limit hits


# ---------- Solver / grader prompts (consistent with paper) ----------

SOLVER_SYSTEM_PROMPT = """You are an expert mathematician solving competition-level problems.
Provide detailed, step-by-step solutions with clear mathematical reasoning.

Requirements:
- Show all your work and intermediate steps
- Justify each major step of your reasoning
- Use proper mathematical notation
- Be thorough but concise
- State your final answer clearly

Solve the problem completely and rigorously."""

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


# ---------- Lazy client builders ----------

_openai_client = None
_anthropic_client = None
_google_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        import httpx
        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=1000)
        timeout = httpx.Timeout(timeout=DEFAULT_BASE_TIMEOUT, connect=30.0,
                                read=DEFAULT_BASE_TIMEOUT, write=30.0)
        _openai_client = AsyncOpenAI(http_client=httpx.AsyncClient(limits=limits, timeout=timeout))
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic()
    return _anthropic_client


def _get_google_client():
    global _google_client
    if _google_client is None:
        from google import genai
        _google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _google_client


# ---------- Per-provider call functions ----------

async def _call_openai(model: str, system: str, user: str,
                       temperature: float, max_tokens: int = 16000) -> dict:
    client = _get_openai_client()
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    # o-series models force temperature=1 and don't accept max_tokens
    if any(p in model.lower() for p in ["o1", "o3", "o4"]):
        api_params.pop("max_tokens", None)
        api_params["temperature"] = 1.0
    else:
        api_params["temperature"] = temperature
        api_params["response_format"] = {"type": "json_object"}
    resp = await client.chat.completions.create(**api_params)
    content = resp.choices[0].message.content or ""
    return {"status": "success", "content": content, "error": None}


async def _call_anthropic(model: str, system: str, user: str,
                          temperature: float, max_tokens: int = 16000) -> dict:
    client = _get_anthropic_client()
    resp = await client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = ""
    if resp.content:
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text
    return {"status": "success", "content": content, "error": None}


async def _call_google(model: str, system: str, user: str,
                       temperature: float, max_tokens: int = 16000) -> dict:
    client = _get_google_client()
    from google.genai.types import GenerateContentConfig
    config = GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=user, config=config,
    )
    content = resp.text or ""
    return {"status": "success", "content": content, "error": None}


# ---------- Unified caller with retries and per-provider semaphore ----------

_provider_sems: dict = {}

def _sem_for(provider: str) -> asyncio.Semaphore:
    if provider not in _provider_sems:
        _provider_sems[provider] = asyncio.Semaphore(PER_PROVIDER_CONCURRENCY[provider])
    return _provider_sems[provider]


async def call_model(model_short: str, system: str, user: str,
                     temperature: float = 0.0, max_tokens: int = 16000,
                     retries: int = DEFAULT_RETRIES) -> dict:
    """Call any supported model by short alias. Includes retries."""
    if model_short == GRADER_MODEL:
        provider = GRADER_PROVIDER
        api_model = GRADER_MODEL
    else:
        provider = SOLVER_PROVIDERS[model_short]
        api_model = API_MODEL_NAMES[model_short]
    sem = _sem_for(provider)

    async with sem:
        last_err = None
        for attempt in range(retries):
            try:
                if provider == "openai":
                    return await _call_openai(api_model, system, user, temperature, max_tokens)
                elif provider == "anthropic":
                    return await _call_anthropic(api_model, system, user, temperature, max_tokens)
                elif provider == "google":
                    return await _call_google(api_model, system, user, temperature, max_tokens)
                else:
                    return {"status": "failed", "content": "",
                            "error": f"unknown provider {provider}"}
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Longer backoff for rate-limit-style errors so the per-minute
                # window has time to refill.
                if "rate_limit" in err_str or "429" in err_str or "rate limit" in err_str:
                    await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS + random.random() * 10)
                else:
                    await asyncio.sleep(min(2 ** attempt + random.random(), 30))
        return {"status": "failed", "content": "",
                "error": f"{type(last_err).__name__}: {str(last_err)[:300]}"}


# ---------- High-level helpers ----------

async def solve(model_short: str, problem_user_msg: str) -> dict:
    """Run the solver. The user message already contains problem + any prefix."""
    return await call_model(model_short, SOLVER_SYSTEM_PROMPT, problem_user_msg, temperature=0.0)


async def grade(problem_type: str, problem_statement: str,
                solution: str, reference_solution: str) -> dict:
    """Run the grader (gpt-4o)."""
    if problem_type == "proof":
        sys = PROOF_GRADER_SYSTEM_PROMPT
        tmpl = PROOF_GRADER_USER_TEMPLATE
    else:
        sys = CALCULATION_GRADER_SYSTEM_PROMPT
        tmpl = CALCULATION_GRADER_USER_TEMPLATE
    user = tmpl.format(problem_statement=problem_statement,
                       solution=solution,
                       reference_solution=reference_solution)
    return await call_model(GRADER_MODEL, sys, user, temperature=0.0)


def parse_solution(content: str) -> dict:
    """Parse JSON {solution, final_answer} from model output, with tolerance."""
    if not content:
        return {"solution": "", "final_answer": "", "_parse_error": "empty"}
    try:
        d = json.loads(content)
        return {"solution": d.get("solution", ""),
                "final_answer": d.get("final_answer", ""),
                "_parse_error": None}
    except Exception:
        # Try to extract a JSON object substring
        import re
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                return {"solution": d.get("solution", ""),
                        "final_answer": d.get("final_answer", ""),
                        "_parse_error": None}
            except Exception as e:
                return {"solution": content, "final_answer": "",
                        "_parse_error": f"json parse: {e}"}
        return {"solution": content, "final_answer": "",
                "_parse_error": "no JSON object found"}


def parse_grade(content: str) -> dict:
    """Parse JSON grade output."""
    if not content:
        return {"grade": "INCORRECT", "_parse_error": "empty"}
    try:
        d = json.loads(content)
        # Normalize grade
        g = (d.get("grade") or "").strip().upper()
        return {
            "grade": g if g in ("CORRECT", "INCORRECT") else "INCORRECT",
            "final_answer_correct": d.get("final_answer_correct"),
            "detailed_feedback": d.get("detailed_feedback", ""),
            "_parse_error": None,
        }
    except Exception:
        import re
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                g = (d.get("grade") or "").strip().upper()
                return {
                    "grade": g if g in ("CORRECT", "INCORRECT") else "INCORRECT",
                    "final_answer_correct": d.get("final_answer_correct"),
                    "detailed_feedback": d.get("detailed_feedback", ""),
                    "_parse_error": None,
                }
            except Exception as e:
                return {"grade": "INCORRECT", "_parse_error": f"json parse: {e}"}
        return {"grade": "INCORRECT", "_parse_error": "no JSON object found"}


# ---------- Standalone health check ----------

async def _health_check():
    print("Running health checks ...")
    msg = ('Reply with JSON {"status": "ok"} only.')
    for short in ["gpt-4o-mini", "claude-sonnet-4", "gemini-2.5-flash"]:
        r = await call_model(short, "You are a test. Reply only the requested JSON.",
                             msg, temperature=0.0, max_tokens=200, retries=2)
        print(f"  {short}: {r['status']} - {r['content'][:200]!r}  err={r['error']}")
    # Grader
    r = await call_model(GRADER_MODEL, "You are a test.", msg, temperature=0.0,
                         max_tokens=200, retries=2)
    print(f"  {GRADER_MODEL} (grader): {r['status']} - {r['content'][:200]!r}  err={r['error']}")


if __name__ == "__main__":
    asyncio.run(_health_check())
