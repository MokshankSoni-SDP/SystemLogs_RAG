"""
Scan Operations Module.
Deterministic Python operations on preprocessed logs for aggregation, counting, and sequencing.
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from .error_taxonomy import detect_error_type, is_error_like


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


def count_error_type(logs: List[Dict], error_type: str) -> Dict[str, int]:
    """
    Count occurrences of a specific semantic error type.
    
    This uses rule-based pattern matching on message content,
    not log levels. Supports all error types from ERROR_SIGNATURES
    plus "unknown_error" fallback.
    
    Args:
        logs: List of preprocessed log records
        error_type: Semantic error category (e.g., "disk_full", "out_of_memory")
        
    Returns:
        Dictionary with counts: {"matched": N, "total_errors": M}
    """
    matched_count = 0
    total_errors = 0
    
    for log in logs:
        detected_type = detect_error_type(log)
        
        if detected_type is not None:
            total_errors += 1
            
            if detected_type == error_type:
                matched_count += 1
    
    return {
        "matched": matched_count,
        "total_errors": total_errors,
        "error_type": error_type
    }


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


def sort_by_time(logs: List[Dict], order: str = "asc") -> List[Dict]:
    """
    Sort logs chronologically by timestamp.
    
    Args:
        logs: List of preprocessed log records
        order: Sort order - "asc" (earliest first) or "desc" (latest first)
        
    Returns:
        Sorted list of log records
    """
    # Parse timestamps
    for log in logs:
        if isinstance(log.get("timestamp"), str):
            try:
                log["_parsed_ts"] = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            except:
                log["_parsed_ts"] = datetime.min
        else:
            log["_parsed_ts"] = datetime.min
    
    # Sort
    reverse = (order == "desc")
    return sorted(logs, key=lambda x: x.get("_parsed_ts", datetime.min), reverse=reverse)


def get_first_event(logs: List[Dict], level: Optional[str] = None, semantic_errors: bool = True) -> Optional[Dict]:
    """
    Get the earliest event of a specific type.
    
    Args:
        logs: List of preprocessed log records
        level: Optional filter by log level
        semantic_errors: If True and level="ERROR", use semantic error detection
        
    Returns:
        Earliest matching log record or None
    """
    # Apply filtering
    if level == "ERROR" and semantic_errors:
        filtered = [log for log in logs if is_error(log)]
    elif level is not None:
        filtered = [log for log in logs if log.get("level") == level]
    else:
        filtered = logs
    
    if not filtered:
        return None
    
    # Get earliest
    sorted_logs = sort_by_time(filtered, order="asc")
    return sorted_logs[0] if sorted_logs else None


def get_last_event(logs: List[Dict], level: Optional[str] = None, semantic_errors: bool = True) -> Optional[Dict]:
    """
    Get the latest event of a specific type.
    
    Args:
        logs: List of preprocessed log records
        level: Optional filter by log level
        semantic_errors: If True and level="ERROR", use semantic error detection
        
    Returns:
        Latest matching log record or None
    """
    # Apply filtering
    if level == "ERROR" and semantic_errors:
        filtered = [log for log in logs if is_error(log)]
    elif level is not None:
        filtered = [log for log in logs if log.get("level") == level]
    else:
        filtered = logs
    
    if not filtered:
        return None
    
    # Get latest
    sorted_logs = sort_by_time(filtered, order="desc")
    return sorted_logs[0] if sorted_logs else None


def filter_by_time_range(
    logs: List[Dict],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[Dict]:
    """
    Filter logs by time range.
    
    Args:
        logs: List of preprocessed log records
        start_time: Optional start time (ISO format or time like "12:00:00")
        end_time: Optional end time (ISO format or time like "23:59:59")
        
    Returns:
        Filtered list of log records
    """
    if not start_time and not end_time:
        return logs
    
    # Parse time bounds
    start_dt = None
    end_dt = None
    
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        except:
            # Try parsing as time only (HH:MM:SS)
            try:
                from datetime import time
                parts = start_time.split(':')
                start_time_obj = time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
                start_dt = datetime.combine(datetime.min.date(), start_time_obj)
            except:
                pass
    
    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except:
            try:
                from datetime import time
                parts = end_time.split(':')
                end_time_obj = time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
                end_dt = datetime.combine(datetime.min.date(), end_time_obj)
            except:
                pass
    
    # Filter
    filtered = []
    for log in logs:
        timestamp = log.get("timestamp")
        if not timestamp:
            continue
        
        try:
            log_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # If time-only bounds, compare time components only
            if start_dt and start_dt.date() == datetime.min.date():
                log_time = log_dt.time()
                if start_time and log_time < start_dt.time():
                    continue
                if end_time and log_time > end_dt.time():
                    continue
            else:
                # Full datetime comparison
                if start_dt and log_dt < start_dt:
                    continue
                if end_dt and log_dt > end_dt:
                    continue
            
            filtered.append(log)
        except:
            continue
    
    return filtered


def find_peak(bucketed_data: Dict[str, int]) -> Tuple[str, int]:
    """
    Find the peak (highest count) in time-bucketed data.
    
    Args:
        bucketed_data: Dictionary mapping time -> count (from bucket_by_time)
        
    Returns:
        Tuple of (peak_time, peak_count)
    """
    if not bucketed_data:
        return ("", 0)
    
    peak_time = max(bucketed_data.keys(), key=lambda k: bucketed_data[k])
    peak_count = bucketed_data[peak_time]
    
    return (peak_time, peak_count)


def rank_by(data: List[Tuple], limit: Optional[int] = None, reverse: bool = True) -> List[Tuple]:
    """
    Rank and optionally limit results.
    
    Args:
        data: List of (value, count) tuples (from count_occurrences)
        limit: Optional limit on results
        reverse: If True, rank descending (highest first)
        
    Returns:
        Ranked list of tuples
    """
    sorted_data = sorted(data, key=lambda x: x[1], reverse=reverse)
    
    if limit:
        return sorted_data[:limit]
    
    return sorted_data


def filter_by_os(logs: List[Dict], os_hint: str) -> List[Dict]:
    """
    Filter logs by OS type.
    
    Args:
        logs: List of preprocessed log records
        os_hint: OS hint ("linux", "windows", "macos")
        
    Returns:
        Filtered list of log records
    """
    return [log for log in logs if log.get("os_hint", "").lower() == os_hint.lower()]


def exclude_process(logs: List[Dict], process: str) -> List[Dict]:
    """
    Exclude logs from a specific process.
    
    Args:
        logs: List of preprocessed log records
        process: Process name to exclude
        
    Returns:
        Filtered list of log records
    """
    return [log for log in logs if log.get("process") != process and log.get("source") != process]


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
            # Check for semantic error_type parameter
            if "error_type" in params:
                result = count_error_type(logs, params["error_type"])
            else:
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
        
        elif operation == "sort_by_time":
            result = sort_by_time(logs, order=params.get("order", "asc"))
        
        elif operation == "get_first_event":
            result = get_first_event(logs, level=params.get("level"))
        
        elif operation == "get_last_event":
            result = get_last_event(logs, level=params.get("level"))
        
        elif operation == "filter_by_time_range":
            result = filter_by_time_range(
                logs,
                start_time=params.get("start_time"),
                end_time=params.get("end_time")
            )
        
        elif operation == "find_peak":
            # Works on bucketed data from previous step
            result = find_peak(result if isinstance(result, dict) else {})
        
        elif operation == "rank_by":
            # Works on count data from previous step
            result = rank_by(
                result if isinstance(result, list) else [],
                limit=params.get("limit"),
                reverse=params.get("reverse", True)
            )
        
        elif operation == "filter_by_os":
            result = filter_by_os(logs, params.get("os_hint", ""))
        
        elif operation == "exclude_process":
            result = exclude_process(logs, params.get("process", ""))
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return result
