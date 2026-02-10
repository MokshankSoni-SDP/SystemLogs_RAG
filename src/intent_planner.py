"""
Intent Planner Module.
Uses LLM to convert user questions into structured execution plans.
LLM does NOT read logs or answer questions - only plans operations.
"""
import json
from typing import Dict, Optional
from .llm_client import LLMClient


PLANNER_PROMPT_TEMPLATE = """You are a QUERY PLANNER for a log analytics system.

Your job is to convert the user's question into a structured execution plan.
You are NOT allowed to answer the question.
You are NOT allowed to invent data.
You must choose from the supported operations only.

Available operations:
1. list_unique_errors - Get all distinct error messages
2. count_occurrences - Count and rank events by frequency
3. filter_by_process - Filter logs by process/source name
4. get_recent_events - Get N most recent events
5. get_before_after_context - Get surrounding context for a process
6. bucket_by_time - Group events by time windows (hour/day)
7. sort_by_time - Sort events chronologically (asc/desc)
8. get_first_event - Get earliest event of a type
9. get_last_event - Get latest event of a type
10. filter_by_time_range - Filter by start/end time
11. find_peak - Find peak timestamp in time buckets (use after bucket_by_time)
12. rank_by - Rank and limit results (use after count_occurrences)
13. filter_by_os - Filter by OS type (linux/windows/macos)
14. exclude_process - Exclude specific process

Log fields available:
- timestamp (ISO-8601 format)
- level (ERROR, WARN, INFO, UNKNOWN)
- process (or source)
- message (log text)
- os_hint (linux, windows, macos, unknown)

Rules:
- Output ONLY valid JSON
- Use null for unknown/unused fields
- Choose the minimal operation needed
- If multiple steps are needed, output them as an ordered list

User question: "{question}"

Output format (strict JSON):
{{
  "steps": [
    {{
      "operation": "string",
      "parameters": {{
        "level": "ERROR|WARN|INFO|null",
        "process": "string|null",
        "group_by": "message|process|level|null",
        "limit": number|null,
        "time_bucket": "hour|day|null",
        "window_size": number|null,
        "order": "asc|desc|null",
        "start_time": "ISO-8601|HH:MM:SS|null",
        "end_time": "ISO-8601|HH:MM:SS|null",
        "os_hint": "linux|windows|macos|null",
        "reverse": true|false|null,
        "error_type": "string|null"
      }}
    }}
  ]
}}

Special Notes:
- Use "error_type" parameter for semantic error classification (e.g., "disk_full", "out_of_memory")
- Available error types: disk_full, out_of_memory, permission_denied, timeout, connection_failed, 
  authentication_failed, process_crash, service_failure, and more
- Extract error_type from user phrases like "Disk full" → "disk_full"

Examples:

Q: "What all errors occurred?"
A: {{"steps": [{{"operation": "list_unique_errors", "parameters": {{"level": "ERROR", "process": null}}}}]}}

Q: "Which is the most frequent error?"
A: {{"steps": [{{"operation": "count_occurrences", "parameters": {{"level": "ERROR", "group_by": "message", "limit": 5}}}}]}}

Q: "What happened before and after process sshd?"
A: {{"steps": [{{"operation": "get_before_after_context", "parameters": {{"process": "sshd", "window_size": 5}}}}]}}

Q: "Summarize recent 5 errors"
A: {{"steps": [{{"operation": "get_recent_events", "parameters": {{"level": "ERROR", "limit": 5}}}}]}}

Q: "Count errors by process"
A: {{"steps": [{{"operation": "count_occurrences", "parameters": {{"level": "ERROR", "group_by": "process", "limit": null}}}}]}}

Q: "Show me the first error that occurred"
A: {{"steps": [{{"operation": "get_first_event", "parameters": {{"level": "ERROR"}}}}]}}

Q: "When did errors peak?"
A: {{"steps": [{{"operation": "bucket_by_time", "parameters": {{"level": "ERROR", "time_bucket": "hour"}}}}, {{"operation": "find_peak", "parameters": {{}}}}]}}

Q: "Top 3 processes with errors"
A: {{"steps": [{{"operation": "count_occurrences", "parameters": {{"level": "ERROR", "group_by": "process", "limit": null}}}}, {{"operation": "rank_by", "parameters": {{"limit": 3}}}}]}}

Q: "Errors before midnight"
A: {{"steps": [{{"operation": "filter_by_time_range", "parameters": {{"end_time": "00:00:00", "start_time": null}}}}, {{"operation": "list_unique_errors", "parameters": {{}}}}]}}

Q: "How many times did 'Disk full' error occur?"
A: {{"steps": [{{"operation": "count_occurrences", "parameters": {{"error_type": "disk_full"}}}}]}}

Q: "Count out of memory errors"
A: {{"steps": [{{"operation": "count_occurrences", "parameters": {{"error_type": "out_of_memory"}}}}]}}

Now generate the plan for the user's question above.
"""


def generate_execution_plan(question: str, llm_client: LLMClient) -> Dict:
    """
    Use LLM to generate a structured execution plan from a user question.
    
    Args:
        question: User's question
        llm_client: LLM client instance
        
    Returns:
        Structured plan dictionary with steps and parameters
        
    Raises:
        ValueError: If plan generation fails or returns invalid JSON
    """
    prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
    
    try:
        # Use plan_query since this is a planning task
        response = llm_client.plan_query(prompt)
        
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove ```json and closing ```
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
        
        # Parse JSON
        plan = json.loads(response)
        
        # Validate plan structure
        if not isinstance(plan, dict) or "steps" not in plan:
            raise ValueError("Invalid plan structure: missing 'steps' key")
        
        if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
            raise ValueError("Invalid plan structure: 'steps' must be a non-empty list")
        
        # Validate each step
        for i, step in enumerate(plan["steps"]):
            if "operation" not in step:
                raise ValueError(f"Step {i}: missing 'operation' key")
            if "parameters" not in step:
                step["parameters"] = {}  # Add empty parameters if missing
        
        return plan
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\\nResponse: {response}")
    except Exception as e:
        raise ValueError(f"Failed to generate execution plan: {e}")


def validate_plan(plan: Dict) -> bool:
    """
    Validate that a plan has the correct structure.
    
    Args:
        plan: Execution plan dictionary
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(plan, dict):
        return False
    
    if "steps" not in plan or not isinstance(plan["steps"], list):
        return False
    
    valid_operations = {
        "list_unique_errors",
        "count_occurrences",
        "filter_by_process",
        "get_recent_events",
        "get_before_after_context",
        "bucket_by_time",
        "sort_by_time",
        "get_first_event",
        "get_last_event",
        "filter_by_time_range",
        "find_peak",
        "rank_by",
        "filter_by_os",
        "exclude_process"
    }
    
    for step in plan["steps"]:
        if not isinstance(step, dict):
            return False
        
        if "operation" not in step:
            return False
        
        if step["operation"] not in valid_operations:
            return False
        
        if "parameters" not in step:
            return False
    
    return True
