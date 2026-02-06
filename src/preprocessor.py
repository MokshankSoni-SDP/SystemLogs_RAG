"""
OS-agnostic log preprocessing module.
Parses raw log lines into canonical records with normalized fields.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional
from src.config import IP_PATTERN, PORT_PATTERN, PID_PATTERN, UUID_PATTERN, MEMORY_ADDR_PATTERN


# Common timestamp patterns for different OS log formats
TIMESTAMP_PATTERNS = [
    # ISO-8601: 2024-01-15T10:23:45Z or 2024-01-15 10:23:45
    (re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)'), 
     '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'),
    
    # Syslog format: Jan 15 10:23:45 or Jan  5 10:23:45
    (re.compile(r'([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'), 
     '%b %d %H:%M:%S'),
    
    # Windows Event Viewer: 01/15/2024 10:23:45 AM
    (re.compile(r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)'), 
     '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S'),
    
    # RFC3339: 2024-01-15T10:23:45.123456+00:00
    (re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})'), 
     '%Y-%m-%dT%H:%M:%S.%f%z'),
]

# Log level patterns
LEVEL_PATTERNS = {
    'ERROR': re.compile(r'\b(ERROR|ERR|FATAL|CRITICAL|CRIT)\b', re.IGNORECASE),
    'WARN': re.compile(r'\b(WARN|WARNING|WRN|CAUTION)\b', re.IGNORECASE),
    'INFO': re.compile(r'\b(INFO|INFORMATION|NOTICE|INF)\b', re.IGNORECASE),
}

# OS detection patterns
OS_HINTS = {
    'linux': [
        re.compile(r'\bsystemd\b|\bsyslog\b|\b/var/log\b|\bkernlog\b', re.IGNORECASE),
        re.compile(r'\b(ubuntu|debian|centos|rhel|fedora)\b', re.IGNORECASE),
    ],
    'windows': [
        re.compile(r'\bEventLog\b|\bEvent\s*Viewer\b|Microsoft-Windows', re.IGNORECASE),
        re.compile(r'\bLogName:|Source:|EventID:', re.IGNORECASE),
        re.compile(r'\bCBS\b|\bCSI\b|\bServicing\s+Stack\b', re.IGNORECASE),
    ],
    'macos': [
        re.compile(r'\b/Library/Logs\b|\bcom\.apple\b|\bkernel\[\d+\]', re.IGNORECASE),
        re.compile(r'\bDarwin\b|\bmacOS\b', re.IGNORECASE),
    ],
}


def extract_timestamp(raw_line: str) -> Optional[str]:
    """
    Extract timestamp from log line and convert to ISO-8601 format.
    
    Args:
        raw_line: Raw log line
        
    Returns:
        ISO-8601 formatted timestamp or None if not found
    """
    for pattern, *formats in TIMESTAMP_PATTERNS:
        match = pattern.search(raw_line)
        if match:
            timestamp_str = match.group(1)
            
            # Try each format for this pattern
            for fmt in formats:
                try:
                    # Handle syslog format (no year)
                    if fmt == '%b %d %H:%M:%S':
                        # Add current year
                        current_year = datetime.now().year
                        timestamp_str_with_year = f"{timestamp_str} {current_year}"
                        dt = datetime.strptime(timestamp_str_with_year, f'{fmt} %Y')
                    else:
                        dt = datetime.strptime(timestamp_str, fmt)
                    
                    return dt.isoformat()
                except ValueError:
                    continue
    
    return None


def extract_log_level(raw_line: str) -> str:
    """
    Extract and normalize log level from log line.
    
    Args:
        raw_line: Raw log line
        
    Returns:
        Normalized level: INFO, WARN, ERROR, or UNKNOWN
    """
    for level, pattern in LEVEL_PATTERNS.items():
        if pattern.search(raw_line):
            return level
    
    return "UNKNOWN"


def detect_os_hint(raw_line: str) -> str:
    """
    Detect OS type from log line patterns.
    
    Args:
        raw_line: Raw log line
        
    Returns:
        OS hint: linux, windows, macos, or unknown
    """
    for os_type, patterns in OS_HINTS.items():
        for pattern in patterns:
            if pattern.search(raw_line):
                return os_type
    
    return "unknown"


def extract_source(raw_line: str) -> str:
    """
    Extract process, service, or channel name from log line.
    
    Args:
        raw_line: Raw log line
        
    Returns:
        Source identifier or "unknown"
    """
    # Common patterns for source/process/service
    patterns = [
        # Syslog: process[pid] or process:
        re.compile(r'\b(\w+)\[\d+\]'),
        re.compile(r'\s(\w+):'),
        
        # Bracketed: [process] or [service]
        re.compile(r'\[([a-zA-Z][a-zA-Z0-9_-]+)\]'),
        
        # Windows EventLog
        re.compile(r'Source:\s*([^\s,]+)'),
    ]
    
    for pattern in patterns:
        match = pattern.search(raw_line)
        if match:
            return match.group(1)
    
    return "unknown"


def clean_message(raw_line: str) -> str:
    """
    Clean message by replacing sensitive/noisy data with placeholders.
    
    Args:
        raw_line: Raw log line
        
    Returns:
        Cleaned message text
    """
    message = raw_line
    
    # Replace IPs
    message = IP_PATTERN.sub('<IP>', message)
    
    # Replace ports
    message = PORT_PATTERN.sub(':<PORT>', message)
    
    # Replace PIDs
    message = PID_PATTERN.sub(r'[<PID>]', message)
    
    # Replace UUIDs
    message = UUID_PATTERN.sub('<UUID>', message)
    
    # Replace memory addresses
    message = MEMORY_ADDR_PATTERN.sub('<ADDR>', message)
    
    return message


def parse_log_line(raw_line: str, fallback_timestamp: Optional[str] = None) -> Dict:
    """
    Parse a single raw log line into a canonical record.
    
    Args:
        raw_line: Raw log line
        fallback_timestamp: Timestamp to use if extraction fails (e.g., ingestion time)
        
    Returns:
        Dictionary with canonical fields
    """
    if not raw_line.strip():
        return None
    
    # Extract timestamp and finding where message begins
    timestamp = None
    message_start_idx = 0
    
    # Improved timestamp extraction that returns end index
    for pattern, *formats in TIMESTAMP_PATTERNS:
        match = pattern.search(raw_line)
        if match:
            timestamp_str = match.group(1)
            message_start_idx = match.end()
            
            # Try each format for this pattern
            for fmt in formats:
                try:
                    if fmt == '%b %d %H:%M:%S':
                        current_year = datetime.now().year
                        timestamp_str_with_year = f"{timestamp_str} {current_year}"
                        dt = datetime.strptime(timestamp_str_with_year, f'{fmt} %Y')
                    else:
                        dt = datetime.strptime(timestamp_str, fmt)
                    
                    timestamp = dt.isoformat()
                    break
                except ValueError:
                    continue
            if timestamp:
                break
    
    if timestamp is None:
        timestamp = fallback_timestamp or datetime.now().isoformat()
        message_start_idx = 0
    
    level = extract_log_level(raw_line)
    os_hint = detect_os_hint(raw_line)
    source = extract_source(raw_line)
    
    # Critical Fix: Only clean the message part, preserve timestamp
    timestamp_part = raw_line[:message_start_idx]
    rest_of_line = raw_line[message_start_idx:]
    cleaned_rest = clean_message(rest_of_line)
    
    # Reconstruct the line or just store the cleaned message?
    # Storing combined is better for context
    final_message = timestamp_part + cleaned_rest
    
    return {
        "timestamp": timestamp,
        "host": "unknown",
        "source": source,
        "level": level,
        "message": final_message,  # Contains original timestamp + cleaned message
        "raw": raw_line,
        "os_hint": os_hint,
    }


def parse_container_log_line(raw_line: str, fallback_timestamp: Optional[str] = None) -> Dict:
    """
    Parse a single container log line (Docker/Kubernetes).
    
    Args:
        raw_line: Raw log line
        fallback_timestamp: Timestamp to use if extraction fails
        
    Returns:
        Dictionary with canonical fields plus container-specific fields
    """
    if not raw_line.strip():
        return None

    # Docker format: Timestamp stream payload
    # Example: 2024-02-12T10:43:21.123Z stdout F Application started
    docker_pattern = re.compile(r'^(\S+)\s+(stdout|stderr)\s+(?:F|P)\s+(.*)$')
    
    # Kubernetes format: Lmmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # Example: E0212 10:43:22.501 controller.go:114 Failed to sync
    k8s_pattern = re.compile(r'^([IWEF])(\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(\S+)\s+(.*)$')

    timestamp = None
    stream = "unknown"
    level = "UNKNOWN"
    message = raw_line
    container_hint = ""

    # Try Docker
    docker_match = docker_pattern.match(raw_line)
    if docker_match:
        ts_str, stream_val, msg_val = docker_match.groups()
        stream = stream_val
        message = msg_val
        
        # Try to parse ISO timestamp
        try:
            timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).isoformat()
        except:
            pass

    # Try Kubernetes if not Docker
    if not timestamp:
        k8s_match = k8s_pattern.match(raw_line)
        if k8s_match:
            severity_char, ts_part, source_part, msg_val = k8s_match.groups()
            
            # Map severity char to level
            severity_map = {'I': 'INFO', 'W': 'WARN', 'E': 'ERROR', 'F': 'FATAL'}
            level = severity_map.get(severity_char, 'UNKNOWN')
            
            # Construct approximate timestamp (K8s logs often lack year, assume current)
            # Format: mmdd hh:mm:ss.uuuuuu
            try:
                current_year = datetime.now().year
                dt = datetime.strptime(f"{current_year}{ts_part}", "%Y%m%d %H:%M:%S.%f")
                timestamp = dt.isoformat()
            except:
                pass
                
            message = f"{source_part} {msg_val}"
            container_hint = "k8s"

    # Fallback to generic extraction if specific formats failed
    if not timestamp:
        # Use existing generic extraction logic
        for pattern, *formats in TIMESTAMP_PATTERNS:
            match = pattern.search(raw_line)
            if match:
                timestamp_str = match.group(1)
                for fmt in formats:
                    try:
                        if fmt == '%b %d %H:%M:%S':
                            timestamp_str_with_year = f"{timestamp_str} {datetime.now().year}"
                            dt = datetime.strptime(timestamp_str_with_year, f'{fmt} %Y')
                        else:
                            dt = datetime.strptime(timestamp_str, fmt)
                        timestamp = dt.isoformat()
                        break
                    except ValueError:
                        continue
                if timestamp:
                    break
    
    if not timestamp:
        timestamp = fallback_timestamp or datetime.now().isoformat()

    # Determine level if not set by K8s logic
    if level == "UNKNOWN":
        if stream == "stderr":
            level = "ERROR" # Often safe assumption for stderr
        else:
            level = extract_log_level(message)

    # Clean message (IPs, PIDs, etc.)
    cleaned_message = clean_message(message)

    return {
        "timestamp": timestamp,
        "level": level,
        "source": "container",
        "stream": stream,
        "message": cleaned_message,
        "raw": raw_line,
        "container_hint": container_hint,
        "log_type": "container"
    }


def preprocess_logs(log_text: str, log_type: str = "system") -> List[Dict]:
    """
    Preprocess entire log file into canonical records.
    
    Args:
        log_text: Raw log content as string
        log_type: "system" or "container"
        
    Returns:
        List of canonical record dictionaries
    """
    lines = log_text.split('\n')
    records = []
    fallback_timestamp = datetime.now().isoformat()
    
    print(f"Preprocessing {len(lines)} lines with mode: {log_type}")

    for line in lines:
        if not line.strip():
            continue
        
        record = None
        if log_type == "container":
            record = parse_container_log_line(line, fallback_timestamp)
        else:
            # System logs (existing logic)
            record = parse_log_line(line, fallback_timestamp)
            if record:
                record["log_type"] = "system"
        
        if record:
            records.append(record)
    
    return records
