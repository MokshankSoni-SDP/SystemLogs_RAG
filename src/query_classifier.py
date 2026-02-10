"""
Query classification module.
Two-stage routing: Hard keyword filters + LLM intent classifier.
"""
import json
from typing import Literal, Optional


# Expanded scan-mode keywords (4 critical categories)
SCAN_MODE_KEYWORDS = [
    # A. Counting / Quantification
    "how many", "count", "number of", "total", "frequency", "occurrences",
    "most", "least", "highest", "lowest", "peak", "maximum", "minimum",
    "most frequent", "least frequent", "top", "bottom",
    
    # B. Global Scope Indicators
    "overall", "across the logs", "in the logs", "entire", "all logs",
    "present in", "all", "every", "complete", "everything",
    "list all", "show all", "what errors", "which errors",
    
    # C. Distribution / Trend Indicators
    "over time", "trend", "clustered", "spread", "distribution", "pattern",
    "increase", "decrease", "before vs after", "spike", "drop",
    "time series", "hourly", "daily", "weekly", "across time",
    
    # D. Ordering / Ranking
    "most common", "least common", "highest frequency", "ranked",
    "top errors", "which process", "what process",
]

# Complex RAG keywords (from config, but can override if needed)
COMPLEX_QUERY_KEYWORDS = [
    "why", "how", "explain", "cause", "reason", "impact",
    "relationship", "correlation", "compare", "difference"
]


# LLM_INTENT_PROMPT = """You are an intent classifier for a log analysis system.
# Your task is to decide HOW a question should be answered.
# Do NOT answer the question. Do NOT infer any facts.

# There are ONLY two valid execution modes:

# 1. "RAG"
#    - Use when the question can be answered by retrieving a small number of log chunks
#    - Examples: specific errors, examples, explanations, "why" questions

# 2. "SCAN"
#    - Use when the question requires counting, aggregation, trends, ordering, or full-log analysis
#    - Examples: totals, most frequent, time distribution, summaries, "what errors are present"

# User question:
# "{question}"

# Return ONLY valid JSON in this format:
# {{
#   "execution_mode": "RAG" | "SCAN",
#   "reason": "one short sentence"
# }}
# """
LLM_INTENT_PROMPT="""
You are an INTENT CLASSIFIER for a log analytics system.

Your job is to decide HOW a question must be answered.
You MUST NOT answer the question.
You MUST NOT infer any facts from logs.

There are ONLY two valid execution modes:

────────────────────────
MODE 1: "RAG"
────────────────────────
Use RAG ONLY if:
- The question can be answered by retrieving ONE OR A FEW specific log chunks
- The answer does NOT require scanning the full log history
- The answer does NOT require counting, ordering, ranking, or time comparison

Typical RAG questions:
- "Show an example of an error"
- "Why did process X fail?"
- "Explain this error message"
- "What does this log mean?"

────────────────────────
MODE 2: "SCAN"
────────────────────────
You MUST choose SCAN if the question requires ANY of the following:
- Counting or totals (how many, number of, frequency)
- Ranking or comparison (most, least, first, last, earliest, latest, peak)
- Ordering across time (first occurred, last occurred, before/after)
- Trends or distribution (over time, clustered, spread, increasing)
- Global inspection (all errors, errors present in the logs, overall summary)
- Absence or existence checks (were there any errors, did X ever happen)

IMPORTANT RULE (DO NOT BREAK):
If answering the question requires examining the FULL LOG DATASET
instead of a few retrieved chunks → it is SCAN.

User question:
"{question}"

Think carefully and classify.

Return ONLY valid JSON in this exact format:
{{
  "execution_mode": "RAG" | "SCAN",
  "reason": "One concise sentence explaining why this mode is required"
}}
"""


def classify_with_llm(question: str, llm_client) -> str:
    """
    Use LLM to classify query intent when keywords are ambiguous.
    
    Args:
        question: User's question
        llm_client: LLM client instance
        
    Returns:
        "rag" or "scan"
    """
    prompt = LLM_INTENT_PROMPT.format(question=question)
    
    try:
        # Use plan_query for classification (low temperature, structured output)
        response = llm_client.plan_query(prompt)
        
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
        
        # Parse JSON
        result = json.loads(response)
        mode = result.get("execution_mode", "RAG").upper()
        
        # Map to our internal format
        if mode == "SCAN":
            return "scan"
        else:
            return "rag"  # Will be further classified as simple/complex
    
    except Exception as e:
        print(f"[Classifier] LLM intent classification failed: {e}")
        # Fallback to simple RAG on error
        return "rag"


def classify_query(question: str, llm_client=None) -> Literal["simple", "complex", "scan"]:
    """
    Two-stage query classification:
    1. Hard keyword filters (fast, deterministic)
    2. LLM intent classifier (precise, catches subtle cases)
    
    Args:
        question: User's question string
        llm_client: Optional LLM client for intent classification
        
    Returns:
        "simple", "complex", or "scan"
    """
    question_lower = question.lower()
    
    # ===== STAGE A: Hard Keyword Filters =====
    # Priority 1: Check for scan-mode keywords
    for keyword in SCAN_MODE_KEYWORDS:
        if keyword in question_lower:
            print(f"[Classifier] Hard filter match: '{keyword}' → SCAN")
            return "scan"
    
    # ===== STAGE B: LLM Intent Classifier =====
    # If LLM client is available, use it for ambiguous cases
    if llm_client is not None:
        llm_decision = classify_with_llm(question, llm_client)
        
        if llm_decision == "scan":
            print(f"[Classifier] LLM classified as SCAN")
            return "scan"
        
        # LLM said RAG, now decide simple vs complex
        print(f"[Classifier] LLM classified as RAG")
    
    # ===== STAGE C: RAG Complexity Classification =====
    # Check for complex RAG keywords
    if len(question_lower.split()) > 12:
        return "complex"
    
    for keyword in COMPLEX_QUERY_KEYWORDS:
        if keyword in question_lower:
            return "complex"
    
    # Default: Simple RAG
    return "simple"
