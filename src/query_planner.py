"""
Query planning module for complex queries.
Uses LLM to generate focused sub-queries for better retrieval.
"""
import json
from typing import List


def clean_sub_queries(sub_queries: List[str]) -> List[str]:
    """
    Post-filter to remove invalid sub-queries that the LLM might generate.
    
    Filters out:
    - Queries with counting/aggregation terms ("how many", "count", "total", "severity")
    - Very short queries (< 3 words)
    
    Args:
        sub_queries: Raw sub-queries from LLM
        
    Returns:
        Cleaned list of valid sub-queries (max 6)
    """
    cleaned = []
    for q in sub_queries:
        q_lower = q.lower()
        # Filter out counting/aggregation queries
        if any(bad in q_lower for bad in ["how many", "count", "total", "severity"]):
            continue
        # Filter out too-short queries
        if len(q.split()) < 3:
            continue
        cleaned.append(q)
    return cleaned[:6]


def generate_sub_queries(question: str, llm_client) -> List[str]:
    """
    Generate focused sub-queries for complex questions using LLM.
    
    Args:
        question: User's complex question
        llm_client: LLM client instance
        
    Returns:
        List of 3-6 focused sub-query strings
    """
    planner_prompt = f"""You are a query planning assistant for a system log retrieval pipeline.

SYSTEM CONTEXT:
- Logs are OS system logs (Linux, Windows, macOS)
- Logs are stored as sequential, time-based chunks
- Each chunk contains ~40 lines (Linux/macOS) or ~25 lines (Windows)
- Chunks are separated by time gaps > 120 seconds
- Retrieval is semantic (vector-based), not keyword search
- The system cannot perform exact counting or aggregation

USER QUESTION:
"{question}"

TASK:
Generate 3 to 6 focused sub-queries that help retrieve log chunks
which together can answer the user question.

RULES:
- Sub-queries must target observable log evidence
- Each sub-query should be answerable by a single chunk
- Do NOT draw conclusions (no "attack", "breach", "incident")
- Do NOT request counting or totals
- Focus on log messages, components, users, services, or time patterns
- Phrase sub-queries like log search intents, not conclusions

GOOD EXAMPLES:
- "failed ssh authentication for root user"
- "sshd authentication failures from same IP"
- "ssh login failures occurring hours apart"

BAD EXAMPLES:
- "how many ssh attacks occurred"
- "evidence of a coordinated attack"
- "severity of the incident"

OUTPUT FORMAT (JSON ONLY):
{{
  "sub_queries": [
    "sub-query 1",
    "sub-query 2",
    "sub-query 3"
  ]
}}

Do not include explanations, markdown, or extra text.
Only output valid JSON."""

    try:
        # Call LLM with the planner prompt using dedicated planning method
        response = llm_client.plan_query(planner_prompt)
        
        # Parse JSON response
        # Try to extract JSON from response (in case LLM adds extra text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            sub_queries = result.get('sub_queries', [])
            
            # Validate and clean
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                # Post-filter to remove invalid queries
                sub_queries = clean_sub_queries(sub_queries)
                # Return cleaned queries, or fallback to original question if all filtered out
                return sub_queries if sub_queries else [question]
        
        # Fallback: if parsing fails, return the original question
        return [question]
        
    except Exception as e:
        # If any error occurs, fall back to using original question
        print(f"Query planning failed: {e}. Using original question.")
        return [question]
