"""
Job management — tracks analysis jobs, their status, and results.
"""

import json
import uuid
import time
import threading
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from image_analyzer import fetch_image, analyze_image
from brands import BRANDS_AND_SKUS, format_q12a
from corrections import find_relevant_corrections, format_corrections_for_prompt
from confidence import compute_confidence
from process import create_thumbnail, build_output, build_client_format, read_raw_data
from rate_limiter import RateLimiter

load_dotenv()

JOBS_FILE = "jobs.json"

# In-memory store for active job threads and progress
_active_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def _load_jobs() -> list[dict]:
    p = Path(JOBS_FILE)
    if not p.exists():
        return []
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_jobs(jobs: list[dict]):
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2, default=str, ensure_ascii=False)


def get_all_jobs() -> list[dict]:
    return _load_jobs()


def get_job(job_id: str) -> dict | None:
    for j in _load_jobs():
        if j["id"] == job_id:
            return j
    return None


def get_job_progress(job_id: str) -> dict:
    """Get real-time progress for an active job."""
    with _job_lock:
        if job_id in _active_jobs:
            return _active_jobs[job_id].get("progress", {})
    return {}


def create_job(
    file_path: str,
    file_name: str,
    model: str,
    start_row: int = 3,
    photo_cols: str = "B,C,D",
) -> str:
    """Create a new job record and return its ID."""
    job_id = str(uuid.uuid4())[:8]

    job = {
        "id": job_id,
        "file_name": file_name,
        "file_path": file_path,
        "model": model,
        "start_row": start_row,
        "photo_cols": photo_cols,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "total_outlets": 0,
        "processed_outlets": 0,
        "total_images": 0,
        "brands_found": [],
        "errors": 0,
        "results_file": None,
        "client_format_file": None,
        "error_message": None,
    }

    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    return job_id


def _update_job(job_id: str, updates: dict):
    """Update a job record in the JSON file."""
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j.update(updates)
            break
    _save_jobs(jobs)


def _process_outlet_for_job(row, model, api_keys, correction_context, limiter):
    """Process one outlet with rate limiting."""
    serial = row["serial"]
    urls = row["urls"]

    all_brands = set()
    all_skus = set()
    thumbnails = []
    total_unidentified = 0
    brands_per_image = []
    ai_confidences = []
    error = None

    for url in urls:
        try:
            limiter.wait()  # Rate limit
            image_data, media_type = fetch_image(url)
            thumbnails.append(create_thumbnail(image_data))
            analysis = analyze_image(
                image_data, media_type,
                model=model, api_keys=api_keys,
                correction_context=correction_context,
            )

            if "error" not in analysis:
                img_brands = []
                for entry in analysis.get("brands_found", []):
                    brand = entry.get("brand", "")
                    if brand in BRANDS_AND_SKUS:
                        all_brands.add(brand)
                        img_brands.append(brand)
                    for sku in entry.get("skus", []):
                        if sku:
                            all_skus.add(sku)
                brands_per_image.append(img_brands)
                total_unidentified += analysis.get("unidentified_packs", 0)
                ai_confidences.append(analysis.get("confidence", "medium"))
            else:
                error = analysis["error"]
        except Exception as e:
            error = str(e)

    brands_sorted = sorted(all_brands)
    skus_sorted = sorted(all_skus)

    worst_ai_conf = "high"
    conf_rank = {"low": 0, "medium": 1, "high": 2}
    for c in ai_confidences:
        if conf_rank.get(c, 1) < conf_rank.get(worst_ai_conf, 2):
            worst_ai_conf = c

    conf_result = compute_confidence(
        ai_confidence=worst_ai_conf,
        brands_found=brands_sorted,
        skus_found=skus_sorted,
        unidentified_packs=total_unidentified,
        num_images=len(urls),
        brands_per_image=brands_per_image,
    )

    return {
        "serial": serial,
        "brands": brands_sorted,
        "skus": skus_sorted,
        "thumbnails": thumbnails,
        "unidentified_packs": total_unidentified,
        "confidence": conf_result["level"],
        "confidence_score": conf_result["score"],
        "error": error if not brands_sorted and error else None,
    }


def run_job(job_id: str):
    """Run a job in a background thread."""
    thread = threading.Thread(target=_run_job_worker, args=(job_id,), daemon=True)
    thread.start()


def _run_job_worker(job_id: str):
    """Background worker that processes a job."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    job = get_job(job_id)
    if not job:
        return

    api_keys = {
        "claude": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "fireworks": os.getenv("FIREWORKS_API_KEY", ""),
    }

    model = job["model"]
    limiter = RateLimiter(model)
    max_workers = limiter.get_safe_workers()

    _update_job(job_id, {"status": "running", "started_at": datetime.now().isoformat()})

    try:
        rows = read_raw_data(job["file_path"], start_row=job["start_row"], photo_cols_str=job["photo_cols"])
        total_images = sum(len(r["urls"]) for r in rows)

        _update_job(job_id, {"total_outlets": len(rows), "total_images": total_images})

        # Init progress
        with _job_lock:
            _active_jobs[job_id] = {
                "progress": {"completed": 0, "total": len(rows), "current_serial": "", "results": []}
            }

        # Load corrections
        recent_corrections = find_relevant_corrections(limit=5)
        correction_context = format_corrections_for_prompt(recent_corrections)

        results = [None] * len(rows)
        completed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, row in enumerate(rows):
                future = executor.submit(
                    _process_outlet_for_job, row, model, api_keys, correction_context, limiter
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    result = future.result()
                    results[idx] = result
                    if result.get("error"):
                        errors += 1
                except Exception as e:
                    results[idx] = {
                        "serial": rows[idx]["serial"],
                        "brands": [], "skus": [], "thumbnails": [],
                        "unidentified_packs": 0, "confidence": "low",
                        "confidence_score": 0, "error": str(e),
                    }
                    errors += 1

                # Update progress
                with _job_lock:
                    if job_id in _active_jobs:
                        _active_jobs[job_id]["progress"]["completed"] = completed
                        _active_jobs[job_id]["progress"]["current_serial"] = str(results[idx]["serial"]) if results[idx] else ""

                _update_job(job_id, {"processed_outlets": completed, "errors": errors})

        # Save results files
        all_brands = set()
        for r in results:
            if r:
                all_brands.update(r.get("brands", []))

        output_dir = Path("job_outputs")
        output_dir.mkdir(exist_ok=True)

        detailed_path = str(output_dir / f"{job_id}_detailed.xlsx")
        client_path = str(output_dir / f"{job_id}_client.xlsx")

        build_output(results, detailed_path)
        build_client_format(results, client_path)

        _update_job(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "brands_found": sorted(all_brands),
            "results_file": detailed_path,
            "client_format_file": client_path,
        })

    except Exception as e:
        _update_job(job_id, {
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e),
        })

    finally:
        with _job_lock:
            _active_jobs.pop(job_id, None)


def delete_job(job_id: str):
    """Delete a job and its output files."""
    job = get_job(job_id)
    if job:
        for key in ["results_file", "client_format_file"]:
            path = job.get(key)
            if path and Path(path).exists():
                Path(path).unlink()

    jobs = _load_jobs()
    jobs = [j for j in jobs if j["id"] != job_id]
    _save_jobs(jobs)
