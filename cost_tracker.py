"""
API usage cost tracking — tracks calls per job and estimates cost.
"""

import json
import threading
from pathlib import Path
from datetime import datetime

COST_LOG_FILE = "cost_log.json"
_cost_lock = threading.Lock()

# Cost per API call by model
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


def _load_cost_log() -> dict:
    p = Path(COST_LOG_FILE)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cost_log(data: dict):
    with open(COST_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def track_call(job_id: str, model: str, call_type: str):
    """
    Record a single API call.

    Args:
        job_id: The job that triggered this call.
        model: Model name (e.g. "gemini-2.5-pro").
        call_type: One of "analysis", "ocr", "sku_refinement".
    """
    cost = COST_PER_CALL.get(model, 0.005)

    with _cost_lock:
        log = _load_cost_log()

        if job_id not in log:
            log[job_id] = {
                "model": model,
                "calls": [],
                "total_calls": 0,
                "total_cost": 0.0,
            }

        entry = {
            "model": model,
            "call_type": call_type,
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
        }

        log[job_id]["calls"].append(entry)
        log[job_id]["total_calls"] += 1
        log[job_id]["total_cost"] = round(log[job_id]["total_cost"] + cost, 6)

        _save_cost_log(log)


def get_job_cost(job_id: str) -> dict:
    """
    Return cost summary for a single job.

    Returns:
        {"model": str, "total_calls": int, "total_cost": float,
         "by_type": {"analysis": int, "ocr": int, ...}}
    """
    log = _load_cost_log()
    entry = log.get(job_id)
    if not entry:
        return {"model": "", "total_calls": 0, "total_cost": 0.0, "by_type": {}}

    by_type: dict[str, int] = {}
    for call in entry.get("calls", []):
        ct = call.get("call_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1

    return {
        "model": entry.get("model", ""),
        "total_calls": entry.get("total_calls", 0),
        "total_cost": entry.get("total_cost", 0.0),
        "by_type": by_type,
    }


def get_total_cost() -> dict:
    """
    Return aggregate cost across all jobs.

    Returns:
        {"total_calls": int, "total_cost": float, "by_model": {model: cost}, "by_job": {job_id: cost}}
    """
    log = _load_cost_log()

    total_calls = 0
    total_cost = 0.0
    by_model: dict[str, float] = {}
    by_job: dict[str, float] = {}

    for job_id, entry in log.items():
        job_cost = entry.get("total_cost", 0.0)
        job_calls = entry.get("total_calls", 0)
        model = entry.get("model", "unknown")

        total_calls += job_calls
        total_cost += job_cost
        by_model[model] = round(by_model.get(model, 0.0) + job_cost, 6)
        by_job[job_id] = job_cost

    return {
        "total_calls": total_calls,
        "total_cost": round(total_cost, 4),
        "by_model": by_model,
        "by_job": by_job,
    }
