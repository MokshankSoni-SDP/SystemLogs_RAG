"""
Time-aware log chunking module with OS-specific boundaries.
Splits preprocessed log records into chunks based on line count, time gaps, and structural boundaries.
"""
import uuid
from typing import List, Dict
from datetime import datetime
from src.config import CHUNK_SIZE_LINUX, CHUNK_SIZE_WINDOWS, TIME_GAP_THRESHOLD_SECONDS


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO-8601 timestamp string to datetime object."""
    try:
        # Handle various ISO-8601 formats
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str)
    except:
        return datetime.now()


def get_time_gap_seconds(ts1: str, ts2: str) -> float:
    """Calculate time gap in seconds between two timestamps."""
    try:
        dt1 = parse_timestamp(ts1)
        dt2 = parse_timestamp(ts2)
        return abs((dt2 - dt1).total_seconds())
    except:
        return 0.0


def detect_multiline_event(records: List[Dict], start_idx: int, max_lookahead: int = 10) -> int:
    """
    Detect if a log record is part of a multi-line event (e.g., stack trace).
    
    Args:
        records: List of preprocessed log records
        start_idx: Starting index to check
        max_lookahead: Maximum lines to look ahead
        
    Returns:
        Index of the last line in the multi-line event
    """
    # Stack trace indicators
    stack_trace_patterns = [
        'at ', 'File "', 'Traceback', '    at', 'Caused by:', 
        'Exception', 'Error:', '  File', 'line '
    ]
    
    end_idx = start_idx
    
    for i in range(start_idx + 1, min(start_idx + max_lookahead, len(records))):
        message = records[i].get('message', '')
        
        # Check if this line looks like a continuation
        is_continuation = False
        for pattern in stack_trace_patterns:
            if pattern in message:
                is_continuation = True
                break
        
        # Check if line starts with whitespace (indented continuation)
        raw = records[i].get('raw', '')
        if raw and raw[0] in [' ', '\t']:
            is_continuation = True
        
        if is_continuation:
            end_idx = i
        else:
            break
    
    return end_idx


def chunk_preprocessed_logs(records: List[Dict], source_name: str = "unknown") -> List[Dict]:
    """
    Chunk preprocessed log records using time-aware, OS/Container-specific boundaries.
    
    Args:
        records: List of preprocessed canonical records
        source_name: Source file name for metadata
        
    Returns:
        List of chunk dictionaries with enhanced metadata
    """
    if not records:
        return []
    
    # Detect log type and dominant OS
    log_type = records[0].get('log_type', 'system')
    
    os_counts = {}
    for record in records:
        os_hint = record.get('os_hint', 'unknown')
        os_counts[os_hint] = os_counts.get(os_hint, 0) + 1
    
    dominant_os = max(os_counts, key=os_counts.get) if os_counts else 'unknown'
    
    # Set chunk parameters
    if log_type == 'container':
        max_chunk_size = 30  # Smaller chunks for containers
        time_threshold = 60.0 # Tighter time gap
    elif dominant_os == 'windows':
        max_chunk_size = CHUNK_SIZE_WINDOWS
        time_threshold = TIME_GAP_THRESHOLD_SECONDS
    else:
        max_chunk_size = CHUNK_SIZE_LINUX
        time_threshold = TIME_GAP_THRESHOLD_SECONDS
    
    # Structural keywords for container breaks
    container_break_keywords = ["back-off", "killed", "oom", "exit code", "restarting", "exception", "panic"]

    chunks = []
    current_chunk_records = []
    
    idx = 0
    while idx < len(records):
        # Determine lookahead for multiline detection
        lookahead = 50 if dominant_os == 'windows' else 10
        multiline_end = detect_multiline_event(records, idx, max_lookahead=lookahead)
        
        if multiline_end > idx:
            # Add all lines of the multi-line event one by one, respecting limits
            for i in range(idx, multiline_end + 1):
                current_chunk_records.append(records[i])
                
                # Rule 1: Limit check inside the event
                if len(current_chunk_records) >= max_chunk_size:
                    chunk = create_chunk_from_records(current_chunk_records, source_name, dominant_os, log_type)
                    chunks.append(chunk)
                    current_chunk_records = []
            idx = multiline_end + 1
        else:
            # Regular single-line record
            current_chunk_records.append(records[idx])
            idx += 1
        
        # Check if we should close the current chunk
        should_close_chunk = False
        
        # Rule 1: Max size
        if len(current_chunk_records) >= max_chunk_size:
            should_close_chunk = True
        
        # Rule 2: Time gap
        if idx < len(records) and current_chunk_records:
            last_ts = current_chunk_records[-1].get('timestamp')
            next_ts = records[idx].get('timestamp')
            
            if last_ts and next_ts:
                time_gap = get_time_gap_seconds(last_ts, next_ts)
                if time_gap > time_threshold:
                    should_close_chunk = True

        # Rule 3: Container Structural Boundaries (if container mode)
        if log_type == 'container' and idx < len(records) and not should_close_chunk:
            next_record = records[idx]
            last_record = current_chunk_records[-1]
            
            # 3a. Stream switch (stdout -> stderr usually implies issue, maybe keep together? 
            # OR stderr burst -> start new chunk? 
            # Strategy: If we are happily in stdout and suddenly hit stderr, stick it in the current chunk 
            # BUT if we have a big block of stderr, maybe isolate it?
            # Let's simple break if we see "structural keywords" in the NEXT line
            
            next_msg_lower = next_record.get('message', '').lower()
            if any(kw in next_msg_lower for kw in container_break_keywords):
                should_close_chunk = True
                
        # Close chunk if boundary reached
        if should_close_chunk or idx >= len(records):
            if current_chunk_records:
                # Create chunk object
                chunk = create_chunk_from_records(current_chunk_records, source_name, dominant_os, log_type)
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_records = []
    
    return chunks


def create_chunk_from_records(records: List[Dict], source_name: str, os_hint: str, log_type: str = "system") -> Dict:
    """
    Create a chunk object from a list of records.
    
    Args:
        records: List of preprocessed records
        source_name: Source file name
        os_hint: Detected OS type
        log_type: Type of logs (system/container)
        
    Returns:
        Chunk dictionary with metadata
    """
    # Join cleaned messages (NOT raw JSON)
    messages = [r.get('message', r.get('raw', '')) for r in records]
    chunk_text = '\n'.join(messages)
    
    # Add header to text for clarity (Optional, but good for embedding context)
    if log_type == 'container':
        header = f"[Container Logs] Stream: {records[0].get('stream', 'mixed')}\n"
        # chunk_text = header + chunk_text  # Decision: Keep text pure or enriched? 
        # Plan says: "Container logs (stdout/stderr):" header in embedding strategy.
        # Let's add it here to ensure it gets embedded.
        chunk_text = "Container logs (stdout/stderr):\n" + chunk_text
    else:
        chunk_text = f"Logs ({os_hint}):\n" + chunk_text

    # Extract timestamps
    timestamps = [r.get('timestamp') for r in records if r.get('timestamp')]
    start_time = timestamps[0] if timestamps else datetime.now().isoformat()
    end_time = timestamps[-1] if timestamps else start_time
    
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": chunk_text,
        "start_time": start_time,
        "end_time": end_time,
        "line_count": len(records),
        "os_hint": os_hint,
        "log_type": log_type,
        "source_file": source_name,
    }


# Legacy chunking function (kept for backward compatibility)
def chunk_logs(log_text: str, chunk_size: int = 30, overlap: int = 5) -> List[Dict]:
    """
    LEGACY: Split log text into chunks of specified size with overlap.
    This is kept for backward compatibility but should not be used in new code.
    Use chunk_preprocessed_logs() instead.
    """
    lines = log_text.split('\n')
    
    # Remove empty lines at the end
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return []
    
    chunks = []
    chunk_index = 0
    start_idx = 0
    
    while start_idx < len(lines):
        end_idx = min(start_idx + chunk_size, len(lines))
        chunk_lines = lines[start_idx:end_idx]
        
        chunk = {
            "chunk_id": f"chunk_{chunk_index:04d}",
            "text": '\n'.join(chunk_lines),
            "start_line": start_idx + 1,
            "end_line": end_idx
        }
        
        chunks.append(chunk)
        chunk_index += 1
        
        if end_idx >= len(lines):
            break
            
        start_idx = end_idx - overlap
    
    return chunks
