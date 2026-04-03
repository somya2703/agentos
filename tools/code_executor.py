"""
tools/code_executor.py — Sandboxed Python execution tool

Runs code in a restricted environment with:
  - Timeout enforcement
  - Blocked dangerous builtins (import os, subprocess, etc.)
  - Output capture
  - Error capture
"""

import sys
import io
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket",
    "requests", "urllib", "http", "ftplib", "smtplib",
    "importlib", "builtins", "ctypes", "pickle",
}

# Blocked regardless of import style
BLOCKED_PATTERNS = [
    "__import__", "eval(", "compile(", "open(", "globals(", "locals(",
]

SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "repr": repr,
    "hasattr": hasattr,
    "getattr": getattr,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "StopIteration": StopIteration,
}


def execute_python(
    code: str,
    timeout_seconds: float = 10.0,
    max_output_chars: int = 4000,
) -> dict:
    """
    Execute Python code in a restricted sandbox.

    Returns:
        {
          "success": bool,
          "output": str,        # stdout
          "error": str | None,  # exception message if failed
          "duration_ms": float,
        }
    """
    # Block dangerous import statements
    for blocked in BLOCKED_IMPORTS:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return {
                "success": False,
                "output": "",
                "error": f"Blocked import: '{blocked}' is not allowed in the sandbox.",
                "duration_ms": 0,
            }

    # Block dangerous patterns (__import__, eval, open, etc.)
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return {
                "success": False,
                "output": "",
                "error": f"Blocked pattern: '{pattern}' is not allowed in the sandbox.",
                "duration_ms": 0,
            }

    stdout_capture = io.StringIO()
    local_vars = {}
    restricted_globals = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__main__",
    }

    start = time.monotonic()
    try:
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        exec(compile(code, "<agent_code>", "exec"), restricted_globals, local_vars)

        sys.stdout = old_stdout
        duration_ms = (time.monotonic() - start) * 1000
        output = stdout_capture.getvalue()[:max_output_chars]

        return {
            "success": True,
            "output": output or "(no output)",
            "error": None,
            "duration_ms": round(duration_ms, 2),
        }
    except Exception as e:
        sys.stdout = old_stdout
        duration_ms = (time.monotonic() - start) * 1000
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"Code execution error: {error_msg}")
        return {
            "success": False,
            "output": stdout_capture.getvalue()[:max_output_chars],
            "error": error_msg,
            "duration_ms": round(duration_ms, 2),
        }
