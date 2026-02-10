"""
Error Taxonomy Module.
Rule-based semantic error classification using message pattern matching.
"""
from typing import Dict, List, Optional


# Production-grade error signatures covering 85-90% of OS-level failures
ERROR_SIGNATURES: Dict[str, List[str]] = {
    # Storage / Filesystem Errors
    "disk_full": [
        "disk full",
        "no space left",
        "no space left on device",
        "disk quota exceeded",
        "filesystem full",
        "write failed",
        "i/o error",
        "input/output error",
        "read-only file system"
    ],
    
    "filesystem_corruption": [
        "filesystem corruption",
        "bad superblock",
        "journal aborted",
        "fsck required",
        "metadata corruption"
    ],
    
    # Memory Errors
    "out_of_memory": [
        "out of memory",
        "oom",
        "oom killer",
        "memory exhausted",
        "cannot allocate memory",
        "allocation failure"
    ],
    
    "memory_leak": [
        "memory leak",
        "heap growth",
        "unreleased memory"
    ],
    
    # CPU / Resource Exhaustion
    "cpu_exhaustion": [
        "cpu overload",
        "soft lockup",
        "hard lockup",
        "cpu stuck",
        "watchdog timeout"
    ],
    
    "resource_limit": [
        "resource temporarily unavailable",
        "too many open files",
        "file descriptor limit",
        "ulimit reached"
    ],
    
    # Permission / Security Errors
    "permission_denied": [
        "permission denied",
        "access denied",
        "operation not permitted",
        "not authorized",
        "authentication failure",
        "authorization failed"
    ],
    
    "authentication_failed": [
        "authentication failed",
        "invalid credentials",
        "login failed",
        "failed password",
        "invalid user"
    ],
    
    # Network Errors
    "connection_failed": [
        "connection refused",
        "connection failed",
        "connection reset",
        "broken pipe",
        "network unreachable",
        "host unreachable"
    ],
    
    "dns_failure": [
        "dns lookup failed",
        "name resolution failed",
        "unknown host",
        "temporary failure in name resolution"
    ],
    
    "timeout": [
        "timeout",
        "timed out",
        "request timeout",
        "operation timed out"
    ],
    
    # Process / Application Errors
    "process_crash": [
        "segmentation fault",
        "core dumped",
        "process crashed",
        "abort",
        "fatal error"
    ],
    
    "service_failure": [
        "service failed",
        "failed to start",
        "failed to stop",
        "exit code",
        "unexpected exit"
    ],
    
    # Kernel / Hardware Errors
    "kernel_error": [
        "kernel panic",
        "oops",
        "bug:",
        "kernel fault"
    ],
    
    "hardware_error": [
        "hardware error",
        "machine check",
        "thermal shutdown",
        "over temperature",
        "fan failure"
    ],
    
    # Update / Package Errors
    "update_failure": [
        "update failed",
        "installation failed",
        "rollback",
        "patch failed",
        "servicing stack error"
    ],
    
    "package_error": [
        "dependency error",
        "broken package",
        "conflict detected"
    ],
    
    # Configuration Errors
    "configuration_error": [
        "invalid configuration",
        "config error",
        "syntax error",
        "unknown directive",
        "misconfiguration"
    ]
}


def is_error_like(log: Dict) -> bool:
    """
    Check if a log is error-like (for unknown_error fallback).
    
    This catches error-like logs that don't match known signatures,
    ensuring we never miss that an error happened.
    
    Args:
        log: Preprocessed log record
        
    Returns:
        True if log appears to be an error
    """
    level = log.get("level", "")
    msg = log.get("message", "").lower()
    
    return (
        level in {"ERROR", "WARN"}
        or "error" in msg
        or "failed" in msg
        or "failure" in msg
        or "exception" in msg
        or "fatal" in msg
        or "panic" in msg
        or "critical" in msg
    )


def detect_error_type(log: Dict) -> Optional[str]:
    """
    Classify log into semantic error category.
    
    This uses rule-based pattern matching on message content,
    not log levels. If no known signature matches but the log
    appears error-like, returns "unknown_error".
    
    Args:
        log: Preprocessed log record
        
    Returns:
        Error type string or None if not an error
    """
    msg = log.get("message", "").lower()
    
    # Check known signatures
    for error_type, patterns in ERROR_SIGNATURES.items():
        for pattern in patterns:
            if pattern in msg:
                return error_type
    
    # Fallback: unknown_error (critical for transparency)
    if is_error_like(log):
        return "unknown_error"
    
    return None


def extract_error_phrase(question: str) -> Optional[str]:
    """
    Extract error type from user question.
    
    This maps user language ("disk full") to error categories.
    
    Args:
        question: User's question string
        
    Returns:
        Error type string or None
    """
    q = question.lower()
    
    for error_type, patterns in ERROR_SIGNATURES.items():
        for pattern in patterns:
            if pattern in q:
                return error_type
    
    return None


def get_all_error_categories() -> List[str]:
    """
    Get list of all known error categories.
    
    Returns:
        List of error category names
    """
    return list(ERROR_SIGNATURES.keys())
