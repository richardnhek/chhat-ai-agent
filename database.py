"""
SQLite database backend — replaces JSON file storage for corrections, jobs, and cost tracking.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "chhat.db"

_db_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return a module-level SQLite connection (created once, reused)."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
        _init_tables(_connection)
    return _connection


def _init_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            serial TEXT,
            image_url TEXT,
            model_used TEXT,
            correction_type TEXT,
            ai_brands TEXT,
            ai_skus TEXT,
            corrected_brands TEXT,
            corrected_skus TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            model TEXT,
            start_row INTEGER,
            photo_cols TEXT,
            status TEXT,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            total_outlets INTEGER,
            processed_outlets INTEGER,
            total_images INTEGER,
            brands_found TEXT,
            errors INTEGER,
            results_file TEXT,
            client_format_file TEXT,
            results_json_file TEXT,
            error_message TEXT,
            last_processed_serial TEXT
        );

        CREATE TABLE IF NOT EXISTS cost_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            job_id TEXT,
            model TEXT,
            call_type TEXT,
            cost REAL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def load_corrections(path: str | None = None) -> list[dict]:
    """Load all corrections from the database. `path` is accepted for backward compatibility but ignored."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM corrections ORDER BY timestamp ASC").fetchall()
    results = []
    for row in rows:
        d = _row_to_dict(row)
        # Reconstruct nested ai_result / corrected_result dicts expected by callers
        d["ai_result"] = {
            "brands": json.loads(d.pop("ai_brands") or "[]"),
            "skus": json.loads(d.pop("ai_skus") or "[]"),
        }
        d["corrected_result"] = {
            "brands": json.loads(d.pop("corrected_brands") or "[]"),
            "skus": json.loads(d.pop("corrected_skus") or "[]"),
        }
        results.append(d)
    return results


def save_correction(correction: dict, path: str | None = None):
    """Save a single correction to the database. `path` is accepted for backward compatibility but ignored."""
    correction["id"] = str(uuid.uuid4())[:8]
    correction["timestamp"] = datetime.now().isoformat()

    # Auto-classify correction type
    ai_brands = set(correction.get("ai_result", {}).get("brands", []))
    corrected_brands = set(correction.get("corrected_result", {}).get("brands", []))

    if ai_brands == corrected_brands:
        ai_skus = set(correction.get("ai_result", {}).get("skus", []))
        corrected_skus = set(correction.get("corrected_result", {}).get("skus", []))
        if ai_skus == corrected_skus:
            correction["correction_type"] = "no_change"
        else:
            correction["correction_type"] = "sku_corrected"
    else:
        added = corrected_brands - ai_brands
        removed = ai_brands - corrected_brands
        if added and removed:
            correction["correction_type"] = "brand_swap"
        elif added:
            correction["correction_type"] = "brand_added"
        elif removed:
            correction["correction_type"] = "brand_removed"

    with _db_lock:
        conn = _get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO corrections
               (id, timestamp, serial, image_url, model_used, correction_type,
                ai_brands, ai_skus, corrected_brands, corrected_skus, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                correction["id"],
                correction["timestamp"],
                correction.get("serial"),
                correction.get("image_url"),
                correction.get("model_used"),
                correction.get("correction_type"),
                json.dumps(correction.get("ai_result", {}).get("brands", [])),
                json.dumps(correction.get("ai_result", {}).get("skus", [])),
                json.dumps(correction.get("corrected_result", {}).get("brands", [])),
                json.dumps(correction.get("corrected_result", {}).get("skus", [])),
                correction.get("notes"),
            ),
        )
        conn.commit()


def save_corrections_batch(corrections_list: list[dict], path: str | None = None):
    """Save multiple corrections at once. `path` is accepted for backward compatibility but ignored."""
    for c in corrections_list:
        if c.get("correction_type") != "no_change":
            save_correction(c)


def find_relevant_corrections(
    ai_brands: list[str] | None = None,
    limit: int = 5,
    path: str | None = None,
) -> list[dict]:
    """
    Find the most relevant past corrections to inject as few-shot examples.
    Only returns corrections where actual changes were made.
    """
    corrections = load_corrections()
    corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    if not corrections:
        return []

    if not ai_brands:
        corrections.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
        return corrections[:limit]

    ai_set = set(ai_brands)
    scored = []
    for c in corrections:
        ai_result_brands = set(c.get("ai_result", {}).get("brands", []))
        corrected_brands = set(c.get("corrected_result", {}).get("brands", []))
        all_brands_in_correction = ai_result_brands | corrected_brands
        overlap = len(ai_set & all_brands_in_correction)
        scored.append((overlap, c))

    scored.sort(key=lambda x: (x[0], x[1].get("timestamp", "")), reverse=True)
    return [c for _, c in scored[:limit]]


def format_corrections_for_prompt(corrections: list[dict]) -> str:
    """Format corrections as few-shot examples for the prompt."""
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

        ai_brands_str = ", ".join(ai.get("brands", [])) or "None"
        ai_skus_str = ", ".join(ai.get("skus", [])) or "None"
        correct_brands = ", ".join(corrected.get("brands", [])) or "None"
        correct_skus = ", ".join(corrected.get("skus", [])) or "None"

        lines.append(f"Correction {i} ({correction_type}):")
        lines.append(f"  AI detected brands: {ai_brands_str}")
        lines.append(f"  AI detected SKUs: {ai_skus_str}")
        lines.append(f"  CORRECT brands: {correct_brands}")
        lines.append(f"  CORRECT SKUs: {correct_skus}")
        if notes:
            lines.append(f'  Reviewer note: "{notes}"')
        lines.append("")

    return "\n".join(lines)


def get_correction_stats(path: str | None = None) -> dict:
    """Get summary stats about corrections. `path` is accepted for backward compatibility but ignored."""
    corrections = load_corrections()
    real_corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    return {
        "total": len(real_corrections),
        "brand_swaps": len([c for c in real_corrections if c.get("correction_type") == "brand_swap"]),
        "brand_added": len([c for c in real_corrections if c.get("correction_type") == "brand_added"]),
        "brand_removed": len([c for c in real_corrections if c.get("correction_type") == "brand_removed"]),
        "sku_corrected": len([c for c in real_corrections if c.get("correction_type") == "sku_corrected"]),
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

_JOBS_COLUMNS = [
    "id", "file_name", "file_path", "model", "start_row", "photo_cols",
    "status", "created_at", "started_at", "completed_at", "total_outlets",
    "processed_outlets", "total_images", "brands_found", "errors",
    "results_file", "client_format_file", "results_json_file",
    "error_message", "last_processed_serial",
]


def _job_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a job row to a dict, deserialising JSON fields."""
    d = _row_to_dict(row)
    # Deserialise brands_found from JSON string to list
    bf = d.get("brands_found")
    if isinstance(bf, str):
        try:
            d["brands_found"] = json.loads(bf)
        except (json.JSONDecodeError, TypeError):
            d["brands_found"] = []
    elif bf is None:
        d["brands_found"] = []
    return d


def get_all_jobs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [_job_row_to_dict(r) for r in rows]


def get_job(job_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return _job_row_to_dict(row)


def create_job(
    file_path: str,
    file_name: str,
    model: str,
    start_row: int = 3,
    photo_cols: str = "B,C,D",
) -> str:
    """Create a new job record and return its ID."""
    job_id = str(uuid.uuid4())[:8]

    with _db_lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO jobs
               (id, file_name, file_path, model, start_row, photo_cols,
                status, created_at, started_at, completed_at,
                total_outlets, processed_outlets, total_images,
                brands_found, errors, results_file, client_format_file,
                results_json_file, error_message, last_processed_serial)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, file_name, file_path, model, start_row, photo_cols,
                "queued", datetime.now().isoformat(), None, None,
                0, 0, 0,
                json.dumps([]), 0, None, None, None, None, None,
            ),
        )
        conn.commit()
    return job_id


def _update_job(job_id: str, updates: dict):
    """Update a job record in the database."""
    if not updates:
        return

    # Serialise brands_found to JSON if present
    if "brands_found" in updates:
        updates = dict(updates)  # shallow copy to avoid mutating caller
        updates["brands_found"] = json.dumps(updates["brands_found"], default=str)

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [job_id]

    with _db_lock:
        conn = _get_conn()
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
        conn.commit()


def delete_job(job_id: str):
    """Delete a job record from the database."""
    with _db_lock:
        conn = _get_conn()
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()


# Compatibility shims — these are called in jobs.py but no longer needed for
# actual persistence. Keep them so imports don't break during transition.
def _load_jobs() -> list[dict]:
    return get_all_jobs()


def _save_jobs(jobs: list[dict]):
    """No-op: individual operations handle persistence now."""
    pass


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------

# Cost per API call by model (kept here so cost_tracker.py can stay thin)
COST_PER_CALL = {
    "gemini-2.5-pro": 0.005,
    "gemini-2.5-flash": 0.001,
    "gemini-3.1-pro": 0.008,
    "gemini-3-pro": 0.008,
    "gemini-3-flash": 0.001,
    "claude-sonnet-4-6": 0.013,
    "claude-haiku-4-5": 0.003,
    "claude-opus-4-6": 0.07,
    "qwen2.5-vl-72b": 0.002,
    "qwen2.5-vl-7b": 0.0003,
}


def track_call(job_id: str, model: str, call_type: str):
    """Record a single API call."""
    cost = COST_PER_CALL.get(model, 0.005)

    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO cost_log (timestamp, job_id, model, call_type, cost) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), job_id, model, call_type, cost),
        )
        conn.commit()


def get_job_cost(job_id: str) -> dict:
    """Return cost summary for a single job."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT model, call_type, cost FROM cost_log WHERE job_id = ?", (job_id,)
    ).fetchall()

    if not rows:
        return {"model": "", "total_calls": 0, "total_cost": 0.0, "by_type": {}}

    total_cost = 0.0
    by_type: dict[str, int] = {}
    model_name = ""
    for row in rows:
        model_name = row["model"]
        total_cost += row["cost"]
        ct = row["call_type"] or "unknown"
        by_type[ct] = by_type.get(ct, 0) + 1

    return {
        "model": model_name,
        "total_calls": len(rows),
        "total_cost": round(total_cost, 6),
        "by_type": by_type,
    }


def get_total_cost() -> dict:
    """Return aggregate cost across all jobs."""
    conn = _get_conn()
    rows = conn.execute("SELECT job_id, model, cost FROM cost_log").fetchall()

    total_calls = 0
    total_cost = 0.0
    by_model: dict[str, float] = {}
    by_job: dict[str, float] = {}

    for row in rows:
        total_calls += 1
        total_cost += row["cost"]
        m = row["model"] or "unknown"
        jid = row["job_id"]
        by_model[m] = round(by_model.get(m, 0.0) + row["cost"], 6)
        by_job[jid] = round(by_job.get(jid, 0.0) + row["cost"], 6)

    return {
        "total_calls": total_calls,
        "total_cost": round(total_cost, 4),
        "by_model": by_model,
        "by_job": by_job,
    }


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def migrate_json_to_db():
    """
    Import existing JSON data (corrections.json, jobs.json, cost_log.json)
    into the SQLite database.  Safe to run multiple times — uses INSERT OR IGNORE.
    """
    base = Path(__file__).parent

    # --- corrections.json ---
    corrections_path = base / "corrections.json"
    if corrections_path.exists():
        try:
            with open(corrections_path, "r") as f:
                corrections = json.load(f)
            with _db_lock:
                conn = _get_conn()
                for c in corrections:
                    conn.execute(
                        """INSERT OR IGNORE INTO corrections
                           (id, timestamp, serial, image_url, model_used, correction_type,
                            ai_brands, ai_skus, corrected_brands, corrected_skus, notes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            c.get("id", str(uuid.uuid4())[:8]),
                            c.get("timestamp"),
                            c.get("serial"),
                            c.get("image_url"),
                            c.get("model_used"),
                            c.get("correction_type"),
                            json.dumps(c.get("ai_result", {}).get("brands", [])),
                            json.dumps(c.get("ai_result", {}).get("skus", [])),
                            json.dumps(c.get("corrected_result", {}).get("brands", [])),
                            json.dumps(c.get("corrected_result", {}).get("skus", [])),
                            c.get("notes"),
                        ),
                    )
                conn.commit()
            print(f"Migrated {len(corrections)} corrections.")
        except Exception as e:
            print(f"Error migrating corrections: {e}")

    # --- jobs.json ---
    jobs_path = base / "jobs.json"
    if jobs_path.exists():
        try:
            with open(jobs_path, "r") as f:
                jobs = json.load(f)
            with _db_lock:
                conn = _get_conn()
                for j in jobs:
                    conn.execute(
                        """INSERT OR IGNORE INTO jobs
                           (id, file_name, file_path, model, start_row, photo_cols,
                            status, created_at, started_at, completed_at,
                            total_outlets, processed_outlets, total_images,
                            brands_found, errors, results_file, client_format_file,
                            results_json_file, error_message, last_processed_serial)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            j.get("id"),
                            j.get("file_name"),
                            j.get("file_path"),
                            j.get("model"),
                            j.get("start_row", 3),
                            j.get("photo_cols", "B,C,D"),
                            j.get("status"),
                            j.get("created_at"),
                            j.get("started_at"),
                            j.get("completed_at"),
                            j.get("total_outlets", 0),
                            j.get("processed_outlets", 0),
                            j.get("total_images", 0),
                            json.dumps(j.get("brands_found", [])),
                            j.get("errors", 0),
                            j.get("results_file"),
                            j.get("client_format_file"),
                            j.get("results_json_file"),
                            j.get("error_message"),
                            j.get("last_processed_serial"),
                        ),
                    )
                conn.commit()
            print(f"Migrated {len(jobs)} jobs.")
        except Exception as e:
            print(f"Error migrating jobs: {e}")

    # --- cost_log.json ---
    cost_path = base / "cost_log.json"
    if cost_path.exists():
        try:
            with open(cost_path, "r") as f:
                cost_data = json.load(f)
            count = 0
            with _db_lock:
                conn = _get_conn()
                for job_id, entry in cost_data.items():
                    for call in entry.get("calls", []):
                        conn.execute(
                            "INSERT INTO cost_log (timestamp, job_id, model, call_type, cost) VALUES (?, ?, ?, ?, ?)",
                            (
                                call.get("timestamp"),
                                job_id,
                                call.get("model"),
                                call.get("call_type"),
                                call.get("cost", 0.0),
                            ),
                        )
                        count += 1
                conn.commit()
            print(f"Migrated {count} cost log entries.")
        except Exception as e:
            print(f"Error migrating cost log: {e}")


# Ensure tables exist on import
_get_conn()
