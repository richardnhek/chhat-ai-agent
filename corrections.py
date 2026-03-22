"""
Correction loop — stores human corrections and retrieves relevant ones as few-shot examples.

Routes to Supabase when SUPABASE_URL is set, otherwise falls back to local SQLite.
"""

import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("SUPABASE_URL"):
    from supabase_db import (
        load_corrections,
        save_correction,
        save_corrections_batch,
        find_relevant_corrections,
        format_corrections_for_prompt,
        get_correction_stats,
    )
else:
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
