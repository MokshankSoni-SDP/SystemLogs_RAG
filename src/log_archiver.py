import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

def persist_raw_logs(raw_text: str, source_name: str, os_hint: str, log_type: str = "system") -> str:
    """
    Save raw log text to disk with metadata sidecar.
    
    Args:
        raw_text: The full raw text of the log
        source_name: Original filename or source identifier
        os_hint: Detected OS (linux, windows, macos, unknown)
        log_type: Container or system
        
    Returns:
        Path to the saved log file
    """
    # 1. Generate path components
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    
    # 2. Determine folder structure
    # raw_logs/{type}/{os_hint}/{date}/
    base_dir = Path("raw_logs") / log_type / os_hint / date_str
    
    # 3. Create directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Save raw log file
    log_filename = f"{upload_id}.log"
    log_path = base_dir / log_filename
    
    # Use UTF-8 and handle potential encoding errors gracefully if needed, 
    # but strictly we want exact bytes if possible. 
    # Since text is already str here, we write as utf-8.
    log_path.write_text(raw_text, encoding="utf-8")
    
    # 5. Save metadata sidecar
    meta_filename = f"{upload_id}.json"
    meta_path = base_dir / meta_filename
    
    metadata = {
        "upload_id": upload_id,
        "original_filename": source_name,
        "uploaded_at": datetime.utcnow().isoformat(),
        "os_hint": os_hint,
        "log_type": log_type,
        "char_count": len(raw_text),
        "line_count": len(raw_text.splitlines()),
        "saved_path": str(log_path)
    }
    
    meta_path.write_text(json.dumps(metadata, indent=2))
    
    print(f"Archived raw logs to: {log_path}")
    return str(log_path)
