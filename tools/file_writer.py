"""
tools/file_writer.py — Write output files safely
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")


def write_file(filename: str, content: str, subdir: str = "") -> dict:
    """
    Write content to a file in the outputs directory.
    Returns {"success": bool, "path": str, "bytes_written": int}
    """
    target_dir = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filename
    safe_name = Path(filename).name
    output_path = target_dir / safe_name

    try:
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"File written: {output_path} ({len(content)} chars)")
        return {"success": True, "path": str(output_path), "bytes_written": len(content)}
    except Exception as e:
        return {"success": False, "path": "", "bytes_written": 0, "error": str(e)}
