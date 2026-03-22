"""
Supabase database backend — replaces SQLite for production cloud deployment.
Provides the same function signatures as database.py for seamless swap.
"""

import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    """Lazy-init Supabase client."""
    global _client
    if _client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        _client = create_client(url, key)
    return _client


# ═══════════════════════════════════════════════════════════════════════════
# CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_corrections(**kwargs) -> list[dict]:
    """Load all corrections from Supabase."""
    client = _get_client()
    result = client.table("corrections").select("*").order("created_at", desc=True).execute()
    corrections = []
    for row in result.data:
        corrections.append({
            "id": row["id"],
            "timestamp": row["created_at"],
            "serial": row.get("serial", ""),
            "image_url": row.get("image_url", ""),
            "model_used": row.get("model_used", ""),
            "correction_type": row.get("correction_type", ""),
            "ai_result": {
                "brands": row.get("ai_brands", []),
                "skus": row.get("ai_skus", []),
            },
            "corrected_result": {
                "brands": row.get("corrected_brands", []),
                "skus": row.get("corrected_skus", []),
            },
            "notes": row.get("notes", ""),
        })
    return corrections


def save_correction(correction: dict, **kwargs):
    """Save a correction to Supabase."""
    client = _get_client()

    correction_id = correction.get("id", str(uuid.uuid4())[:8])

    # Auto-classify correction type
    ai_brands = set(correction.get("ai_result", {}).get("brands", []))
    corrected_brands = set(correction.get("corrected_result", {}).get("brands", []))
    ai_skus = set(correction.get("ai_result", {}).get("skus", []))
    corrected_skus = set(correction.get("corrected_result", {}).get("skus", []))

    if ai_brands == corrected_brands:
        if ai_skus == corrected_skus:
            correction_type = "no_change"
        else:
            correction_type = "sku_corrected"
    else:
        added = corrected_brands - ai_brands
        removed = ai_brands - corrected_brands
        if added and removed:
            correction_type = "brand_swap"
        elif added:
            correction_type = "brand_added"
        elif removed:
            correction_type = "brand_removed"
        else:
            correction_type = "unknown"

    row = {
        "id": correction_id,
        "serial": str(correction.get("serial", "")),
        "image_url": correction.get("image_url", ""),
        "model_used": correction.get("model_used", ""),
        "correction_type": correction_type,
        "ai_brands": list(correction.get("ai_result", {}).get("brands", [])),
        "ai_skus": list(correction.get("ai_result", {}).get("skus", [])),
        "corrected_brands": list(correction.get("corrected_result", {}).get("brands", [])),
        "corrected_skus": list(correction.get("corrected_result", {}).get("skus", [])),
        "notes": correction.get("notes", ""),
    }

    client.table("corrections").upsert(row).execute()


def save_corrections_batch(corrections_list: list[dict], **kwargs):
    """Save multiple corrections."""
    for c in corrections_list:
        if c.get("correction_type") != "no_change":
            save_correction(c)


def find_relevant_corrections(ai_brands: list[str] | None = None, limit: int = 5, **kwargs) -> list[dict]:
    """Find relevant past corrections for few-shot learning."""
    corrections = load_corrections()
    real_corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    if not real_corrections:
        return []

    if not ai_brands:
        return real_corrections[:limit]

    # Score by brand overlap
    ai_set = set(ai_brands)
    scored = []
    for c in real_corrections:
        ai_result_brands = set(c.get("ai_result", {}).get("brands", []))
        corrected_brands = set(c.get("corrected_result", {}).get("brands", []))
        all_brands = ai_result_brands | corrected_brands
        overlap = len(ai_set & all_brands)
        scored.append((overlap, c))

    scored.sort(key=lambda x: (x[0], x[1].get("timestamp", "")), reverse=True)
    return [c for _, c in scored[:limit]]


def format_corrections_for_prompt(corrections: list[dict]) -> str:
    """Format corrections as few-shot examples."""
    if not corrections:
        return ""

    lines = [
        "LEARNING FROM PAST CORRECTIONS:",
        "The following are real examples where the AI made mistakes that were corrected by human reviewers.",
        "Use these to avoid repeating the same errors:\n",
    ]

    for i, c in enumerate(corrections, 1):
        ai = c.get("ai_result", {})
        corrected = c.get("corrected_result", {})
        notes = c.get("notes", "")
        correction_type = c.get("correction_type", "unknown")

        ai_brands = ", ".join(ai.get("brands", [])) or "None"
        ai_skus = ", ".join(ai.get("skus", [])) or "None"
        correct_brands = ", ".join(corrected.get("brands", [])) or "None"
        correct_skus = ", ".join(corrected.get("skus", [])) or "None"

        lines.append(f"Correction {i} ({correction_type}):")
        lines.append(f"  AI detected brands: {ai_brands}")
        lines.append(f"  AI detected SKUs: {ai_skus}")
        lines.append(f"  CORRECT brands: {correct_brands}")
        lines.append(f"  CORRECT SKUs: {correct_skus}")
        if notes:
            lines.append(f'  Reviewer note: "{notes}"')
        lines.append("")

    return "\n".join(lines)


def get_correction_stats(**kwargs) -> dict:
    """Get correction statistics."""
    corrections = load_corrections()
    real = [c for c in corrections if c.get("correction_type") != "no_change"]
    return {
        "total": len(real),
        "brand_swaps": len([c for c in real if c.get("correction_type") == "brand_swap"]),
        "brand_added": len([c for c in real if c.get("correction_type") == "brand_added"]),
        "brand_removed": len([c for c in real if c.get("correction_type") == "brand_removed"]),
        "sku_corrected": len([c for c in real if c.get("correction_type") == "sku_corrected"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# JOBS
# ═══════════════════════════════════════════════════════════════════════════

def get_all_jobs() -> list[dict]:
    """Get all jobs."""
    client = _get_client()
    result = client.table("jobs").select("*").order("created_at", desc=False).execute()
    return [_row_to_job(r) for r in result.data]


def get_job(job_id: str) -> dict | None:
    """Get a single job by ID."""
    client = _get_client()
    result = client.table("jobs").select("*").eq("id", job_id).execute()
    if result.data:
        return _row_to_job(result.data[0])
    return None


def create_job(file_path: str, file_name: str, model: str,
               start_row: int = 3, photo_cols: str = "B,C,D") -> str:
    """Create a new job and return its ID."""
    client = _get_client()
    job_id = str(uuid.uuid4())[:8]

    row = {
        "id": job_id,
        "file_name": file_name,
        "file_path": file_path,
        "model": model,
        "start_row": start_row,
        "photo_cols": photo_cols,
        "status": "queued",
    }

    client.table("jobs").insert(row).execute()
    return job_id


def _update_job(job_id: str, updates: dict):
    """Update a job."""
    client = _get_client()
    # Convert brands_found list to JSON if present
    row_updates = {}
    for k, v in updates.items():
        if k == "brands_found" and isinstance(v, list):
            row_updates[k] = v  # Supabase handles JSONB
        else:
            row_updates[k] = v
    client.table("jobs").update(row_updates).eq("id", job_id).execute()


def db_delete_job(job_id: str):
    """Delete a job from the database."""
    client = _get_client()
    client.table("cost_log").delete().eq("job_id", job_id).execute()
    client.table("jobs").delete().eq("id", job_id).execute()


def _row_to_job(row: dict) -> dict:
    """Convert a Supabase row to the job dict format expected by the app."""
    return {
        "id": row["id"],
        "file_name": row.get("file_name", ""),
        "file_path": row.get("file_path", ""),
        "model": row.get("model", ""),
        "start_row": row.get("start_row", 3),
        "photo_cols": row.get("photo_cols", "B,C,D"),
        "status": row.get("status", "queued"),
        "created_at": row.get("created_at", ""),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "total_outlets": row.get("total_outlets", 0),
        "processed_outlets": row.get("processed_outlets", 0),
        "total_images": row.get("total_images", 0),
        "brands_found": row.get("brands_found", []),
        "errors": row.get("errors", 0),
        "results_file": row.get("results_file"),
        "client_format_file": row.get("client_format_file"),
        "results_json_file": row.get("results_json_file"),
        "error_message": row.get("error_message"),
        "last_processed_serial": row.get("last_processed_serial"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# COST TRACKING
# ═══════════════════════════════════════════════════════════════════════════

COST_PER_CALL = {
    "gemini-2.5-pro": 0.005,
    "gemini-2.5-flash": 0.001,
    "gemini-3.1-pro": 0.008,
    "gemini-3-pro": 0.008,
    "gemini-3-flash": 0.002,
    "claude-sonnet-4-6": 0.013,
    "claude-haiku-4-5": 0.003,
    "claude-opus-4-6": 0.07,
    "qwen2.5-vl-72b": 0.002,
    "qwen2.5-vl-7b": 0.0003,
}


def track_call(job_id: str, model: str, call_type: str = "analysis"):
    """Track an API call for cost monitoring."""
    client = _get_client()
    cost = COST_PER_CALL.get(model, 0.005)
    client.table("cost_log").insert({
        "job_id": job_id,
        "model": model,
        "call_type": call_type,
        "cost": cost,
    }).execute()


def get_job_cost(job_id: str) -> dict:
    """Get cost summary for a specific job."""
    client = _get_client()
    result = client.table("cost_log").select("*").eq("job_id", job_id).execute()
    rows = result.data

    total_cost = sum(r.get("cost", 0) for r in rows)
    total_calls = len(rows)
    by_type = {}
    for r in rows:
        ct = r.get("call_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + r.get("cost", 0)

    return {"total_cost": total_cost, "total_calls": total_calls, "by_type": by_type}


def get_total_cost() -> dict:
    """Get total cost across all jobs."""
    client = _get_client()
    result = client.table("cost_log").select("*").execute()
    rows = result.data

    total_cost = sum(r.get("cost", 0) for r in rows)
    total_calls = len(rows)
    by_model = {}
    by_job = {}
    for r in rows:
        m = r.get("model", "unknown")
        j = r.get("job_id", "unknown")
        by_model[m] = by_model.get(m, 0) + r.get("cost", 0)
        by_job[j] = by_job.get(j, 0) + r.get("cost", 0)

    return {
        "total_cost": total_cost,
        "total_calls": total_calls,
        "by_model": by_model,
        "by_job": by_job,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MIGRATION
# ═══════════════════════════════════════════════════════════════════════════

def migrate_sqlite_to_supabase():
    """Migrate data from local SQLite to Supabase."""
    try:
        import database as sqlite_db

        # Migrate corrections
        corrections = sqlite_db.load_corrections()
        for c in corrections:
            save_correction(c)
        print(f"Migrated {len(corrections)} corrections")

        # Migrate jobs
        jobs = sqlite_db.get_all_jobs()
        client = _get_client()
        for j in jobs:
            row = {k: v for k, v in j.items() if k != "brands_found"}
            row["brands_found"] = j.get("brands_found", [])
            try:
                client.table("jobs").upsert(row).execute()
            except Exception as e:
                print(f"  Job {j['id']} failed: {e}")
        print(f"Migrated {len(jobs)} jobs")

        print("Migration complete!")
    except Exception as e:
        print(f"Migration error: {e}")
