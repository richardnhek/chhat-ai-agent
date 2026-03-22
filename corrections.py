"""
Correction loop — stores human corrections and retrieves relevant ones as few-shot examples.

Storage backed by SQLite via database.py.
"""

from database import (
    load_corrections,
    save_correction,
    save_corrections_batch,
    find_relevant_corrections,
    format_corrections_for_prompt,
    get_correction_stats,
)

__all__ = [
    "load_corrections",
    "save_correction",
    "save_corrections_batch",
    "find_relevant_corrections",
    "format_corrections_for_prompt",
    "get_correction_stats",
]
