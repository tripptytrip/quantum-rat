
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# In-memory cache for line offsets
# For a production system, this could be backed by a more robust cache (e.g., Redis, Memcached)
# or persisted to disk as .idx files.
OFFSET_CACHE: Dict[Path, List[int]] = {}

def get_run_root() -> Path:
    """Gets the root directory for runs, with a fallback."""
    run_dir_str = os.environ.get("CRITICAL_RAT_RUNS_DIR", "runs")
    return Path(run_dir_str).resolve()

def get_safe_path(run_id: str, *subpaths) -> Path:
    """
    Safely joins a run ID and subpaths to a path within the run root.
    Raises ValueError on path traversal attempts.
    """
    run_root = get_run_root()
    
    # Create the full path
    full_path = run_root.joinpath(run_id, *subpaths)
    
    # Get the absolute, canonical path
    real_path = os.path.realpath(full_path)
    
    # Ensure the real path is inside the run root
    if not os.path.commonpath([run_root, real_path]) == str(run_root):
        raise ValueError(f"Path traversal attempt detected: {full_path}")

    return Path(real_path)

def get_jsonl_offsets(file_path: Path) -> List[int]:
    """
    Gets byte offsets for each line in a JSONL file, using a cache.
    """
    if file_path in OFFSET_CACHE:
        return OFFSET_CACHE[file_path]
    
    offsets = [0]
    with open(file_path, "rb") as f:
        while f.readline():
            offsets.append(f.tell())
    offsets.pop() # Last offset is EOF, not the start of a line
    
    OFFSET_CACHE[file_path] = offsets
    return offsets

def read_jsonl_paged(file_path: Path, start: int, limit: int) -> Tuple[List[Dict], int]:
    """
    Reads a page of data from a JSONL file using an offset index.
    """
    offsets = get_jsonl_offsets(file_path)
    total_lines = len(offsets)
    
    if start >= total_lines:
        return [], total_lines

    lines_to_read = min(limit, total_lines - start)
    
    ticks = []
    with open(file_path, "r") as f:
        f.seek(offsets[start])
        for _ in range(lines_to_read):
            line = f.readline()
            if not line:
                break
            ticks.append(json.loads(line))
            
    return ticks, total_lines
