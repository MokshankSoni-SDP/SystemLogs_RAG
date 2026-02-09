"""
Scan Operations Module.
Deterministic Python operations on preprocessed logs for aggregation, counting, and sequencing.
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime


def is_error(log: Dict) -> bool:
    """
    Check if a log is an error using level OR message content.
    This handles cases where preprocessing sets level="UNKNOWN" but message contains error indicators.
    
    Args:
        log: Preprocessed log record
        
    Returns:
        True if log is an error
    """
    # Check level first
    if log.get("level") == "ERROR":
        return True
    
    # Semantic fallback: check message content
    message = log.get("message", "").lower()
    return any(keyword in message for keyword in [
        "error", "failed", "failure", "exception", "fatal", "critical"
    ])


def list_unique_errors(logs: List[Dict]) -> List[str]:
    """
    Get all unique error messages using semantic detection.
    
    Args:
        logs: List of preprocessed log records
        
    Returns:
        Sorted list of unique error messages
    """
    return sorted(set(
        log["message"]
        for log in logs
        if is_error(log)
    ))


def count_occurrences(
    logs: List[Dict], 
    group_by: str = "message", 
    level: Optional[str] = None, 
    limit: Optional[int] = None,
    semantic_errors: bool = True
) -> List[Tuple[str, int]]:
    """
    Count occurrences of log events, grouped by a specified field.
    
    Args:
        logs: List of preprocessed log records
        group_by: Field to group by ("message", "process", "level")
        level: Optional filter by log level ("ERROR" uses semantic detection if semantic_errors=True)
        limit: Optional limit on number of results
        semantic_errors: If True and level="ERROR", use semantic error detection
        
    Returns:
        List of (value, count) tuples, sorted by frequency (descending)
    """
    # Apply filtering
    if level == "ERROR" and semantic_errors:
        # Use semantic error detection
        filtered = [log for log in logs if is_error(log)]
    elif level is not None:
        # Strict level matching
        filtered = [log for log in logs if log.get("level") == level]
    else:
        filtered = logs
    
    counter = Counter(log.get(group_by, "unknown") for log in filtered)
    return counter.most_common(limit)


def filter_by_process(logs: List[Dict], process: str) -> List[Dict]:
    """
    Filter logs by process name.
    
    Args:
        logs: List of preprocessed log records
        process: Process name to filter by
        
    Returns:
        Filtered list of log records
    """
    return [log for log in logs if log.get("process") == process or log.get("source") == process]


def get_recent_events(
    logs: List[Dict], 
    level: Optional[str] = None, 
    limit: int = 5,
    semantic_errors: bool = True
) -> List[Dict]:
    """
    Get the N most recent log events.
    
    Args:
        logs: List of preprocessed log records
        level: Optional filter by log level ("ERROR" uses semantic detection if semantic_errors=True)
        limit: Number of events to return
        semantic_errors: If True and level="ERROR", use semantic error detection
        
    Returns:
        List of most recent log records
    """
    # Apply filtering
    if level == "ERROR" and semantic_errors:
        filtered = [log for log in logs if is_error(log)]
    elif level is not None:
        filtered = [log for log in logs if log.get("level") == level]
    else:
        filtered = logs
    
    # Parse timestamps and sort
    for log in filtered:
        if isinstance(log.get("timestamp"), str):
            try:
                log["_parsed_ts"] = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            except:
                log["_parsed_ts"] = datetime.min
    
    sorted_logs = sorted(filtered, key=lambda x: x.get("_parsed_ts", datetime.min), reverse=True)
    return sorted_logs[:limit]


def get_before_after_context(
    logs: List[Dict], 
    process: str, 
    window_size: int = 5
) -> List[Dict]:
    """
    Get surrounding context (before and after) for a specific process.
    
    Args:
        logs: List of preprocessed log records
        process: Process name to find
        window_size: Number of lines before and after to include
        
    Returns:
        List of log records with context
    """
    for i, log in enumerate(logs):
        if log.get("process") == process or log.get("source") == process:
            start = max(0, i - window_size)
            end = min(len(logs), i + window_size + 1)
            return logs[start:end]
    return []


def bucket_by_time(
    logs: List[Dict], 
    level: Optional[str] = None, 
    bucket: str = "hour"
) -> Dict[str, int]:
    """
    Group log events by time buckets.
    
    Args:
        logs: List of preprocessed log records
        level: Optional filter by log level
        bucket: Time bucket size ("hour" or "day")
        
    Returns:
        Dictionary mapping time bucket to event count
    """
    buckets = defaultdict(int)
    
    for log in logs:
        if level and log.get("level") != level:
            continue
        
        timestamp = log.get("timestamp")
        if not timestamp:
            continue
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            if bucket == "hour":
                key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
            else:  # day
                key = dt.date().isoformat()
            
            buckets[key] += 1
        except:
            continue
    
    return dict(sorted(buckets.items()))


def execute_plan(logs: List[Dict], plan: Dict) -> Any:
    """
    Execute a structured plan on the logs.
    
    Args:
        logs: List of preprocessed log records
        plan: Execution plan with steps and parameters
        
    Returns:
        Result of the last operation in the plan
    """
    result = None
    
    for step in plan.get("steps", []):
        operation = step.get("operation")
        params = step.get("parameters", {})
        
        if operation == "list_unique_errors":
            result = list_unique_errors(logs)
        
        elif operation == "count_occurrences":
            result = count_occurrences(
                logs,
                group_by=params.get("group_by", "message"),
                level=params.get("level"),
                limit=params.get("limit")
            )
        
        elif operation == "filter_by_process":
            result = filter_by_process(logs, params.get("process", ""))
        
        elif operation == "get_recent_events":
            result = get_recent_events(
                logs,
                level=params.get("level"),
                limit=params.get("limit", 5)
            )
        
        elif operation == "get_before_after_context":
            result = get_before_after_context(
                logs,
                process=params.get("process", ""),
                window_size=params.get("window_size", 5)
            )
        
        elif operation == "bucket_by_time":
            result = bucket_by_time(
                logs,
                level=params.get("level"),
                bucket=params.get("time_bucket", "hour")
            )
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return result
