"""
API usage cost tracking — tracks calls per job and estimates cost.

Routes to Supabase when SUPABASE_URL is set, otherwise falls back to local SQLite.
"""

import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("SUPABASE_URL"):
    from supabase_db import (
        COST_PER_CALL,
        track_call,
        get_job_cost,
        get_total_cost,
    )
else:
    from database import (
        COST_PER_CALL,
        track_call,
        get_job_cost,
        get_total_cost,
    )

__all__ = [
    "COST_PER_CALL",
    "track_call",
    "get_job_cost",
    "get_total_cost",
]
