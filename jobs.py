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
from enhancements import analyze_image_enhanced, compute_blur_score
from rate_limiter import RateLimiter
from cost_tracker import track_call
from database import (
    get_all_jobs,
    get_job,
    create_job,
    _update_job,
    delete_job as _db_delete_job,
)

load_dotenv()

PARTIAL_DIR = Path("job_outputs")

# In-memory store for active job threads and progress
_active_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def get_job_progress(job_id: str) -> dict:
    """Get real-time progress for an active job."""
    with _job_lock:
        if job_id in _active_jobs:
            return _active_jobs[job_id].get("progress", {})
    return {}


def _save_partial_results(job_id: str, results: list[dict | None]):
    """Save partial results to disk for crash recovery."""
    PARTIAL_DIR.mkdir(exist_ok=True)
    partial_path = PARTIAL_DIR / f"{job_id}_partial.json"
    # Filter out None entries and save with their index
    serializable = []
    for idx, r in enumerate(results):
        if r is not None:
            # Thumbnails are bytes — drop them from partial saves
            entry = {k: v for k, v in r.items() if k != "thumbnails"}
            entry["_idx"] = idx
            serializable.append(entry)
    with open(partial_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)


def _load_partial_results(job_id: str) -> list[dict]:
    """Load partial results from disk. Returns list of dicts with _idx field."""
    partial_path = PARTIAL_DIR / f"{job_id}_partial.json"
    if not partial_path.exists():
        return []
    try:
        with open(partial_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _cleanup_partial(job_id: str):
    """Remove partial results file after successful completion."""
    partial_path = PARTIAL_DIR / f"{job_id}_partial.json"
    if partial_path.exists():
        partial_path.unlink()


def _process_outlet_for_job(row, model, api_keys, correction_context, limiter):
    """Process one outlet with rate limiting. Returns result dict with per-image details."""
    serial = row["serial"]
    urls = row["urls"]

    all_brands = set()
    all_skus = set()
    thumbnails = []
    total_unidentified = 0
    brands_per_image = []
    ai_confidences = []
    per_image_details = []
    error = None
    api_calls = 0

    for url in urls:
        try:
            limiter.wait()  # Rate limit
            image_data, media_type = fetch_image(url)
            thumbnails.append(create_thumbnail(image_data))

            analysis = analyze_image_enhanced(
                image_data, media_type,
                model=model, api_keys=api_keys,
                correction_context=correction_context,
                enable_enhancement=True,
                enable_ocr=True,
                enable_sku_refinement=True,
            )
            api_calls += 1

            img_detail = {
                "url": url,
                "blur_score": analysis.get("blur_score", round(compute_blur_score(image_data), 1)),
                "image_enhanced": analysis.get("image_enhanced", False),
            }

            if "error" not in analysis:
                img_brands = []
                img_skus = []
                brand_entries = []
                for entry in analysis.get("brands_found", []):
                    brand = entry.get("brand", "")
                    if brand in BRANDS_AND_SKUS:
                        all_brands.add(brand)
                        img_brands.append(brand)
                    for sku in entry.get("skus", []):
                        if sku:
                            all_skus.add(sku)
                            img_skus.append(sku)
                    brand_entries.append({
                        "brand": entry.get("brand", ""),
                        "skus": entry.get("skus", []),
                        "notes": entry.get("notes", ""),
                    })
                brands_per_image.append(img_brands)
                total_unidentified += analysis.get("unidentified_packs", 0)
                ai_confidences.append(analysis.get("confidence", "medium"))
                img_detail["brands_found"] = brand_entries
                img_detail["brands"] = img_brands
                img_detail["skus"] = img_skus
                img_detail["confidence"] = analysis.get("confidence", "medium")
                img_detail["unidentified_packs"] = analysis.get("unidentified_packs", 0)
                img_detail["notes"] = analysis.get("notes", "")
            else:
                img_detail["error"] = analysis["error"]
                error = analysis["error"]

            per_image_details.append(img_detail)
        except Exception as e:
            per_image_details.append({"url": url, "error": str(e)})
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
        "confidence_factors": conf_result["factors"],
        "error": error if not brands_sorted and error else None,
        "api_calls": api_calls,
        "urls": urls,
        "per_image": per_image_details,
    }


def run_job(job_id: str):
    """Run a job in a background thread."""
    thread = threading.Thread(target=_run_job_worker, args=(job_id,), daemon=True)
    thread.start()


def _run_job_worker(job_id: str, skip_serials: set | None = None):
    """Background worker that processes a job.

    Args:
        job_id: The job to process.
        skip_serials: If resuming, set of serial numbers already processed.
    """
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

        # If resuming, pre-fill results from partial data
        already_done = 0
        if skip_serials:
            partial_data = _load_partial_results(job_id)
            for entry in partial_data:
                idx = entry.pop("_idx", None)
                if idx is not None and idx < len(results):
                    entry["thumbnails"] = []  # Thumbnails not saved in partial
                    results[idx] = entry
                    already_done += 1

        completed = already_done
        errors = job.get("errors", 0) if skip_serials else 0

        # Determine which rows still need processing
        rows_to_process = []
        for idx, row in enumerate(rows):
            if skip_serials and str(row["serial"]) in skip_serials:
                continue
            rows_to_process.append((idx, row))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, row in rows_to_process:
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

                    # Track API cost for this outlet
                    num_calls = result.get("api_calls", 1)
                    for _ in range(num_calls):
                        track_call(job_id, model, "analysis")

                except Exception as e:
                    results[idx] = {
                        "serial": rows[idx]["serial"],
                        "brands": [], "skus": [], "thumbnails": [],
                        "unidentified_packs": 0, "confidence": "low",
                        "confidence_score": 0, "error": str(e),
                        "api_calls": 0,
                    }
                    errors += 1

                # Save partial results incrementally
                _save_partial_results(job_id, results)

                # Update progress
                serial_str = str(results[idx]["serial"]) if results[idx] else ""
                with _job_lock:
                    if job_id in _active_jobs:
                        _active_jobs[job_id]["progress"]["completed"] = completed
                        _active_jobs[job_id]["progress"]["current_serial"] = serial_str

                _update_job(job_id, {
                    "processed_outlets": completed,
                    "errors": errors,
                    "last_processed_serial": serial_str,
                })

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

        # Save raw JSON results for the Outlet Detail page
        json_results_path = str(output_dir / f"{job_id}_results.json")
        json_serializable = []
        for r in results:
            if r is not None:
                entry = {k: v for k, v in r.items() if k != "thumbnails"}
                json_serializable.append(entry)
            else:
                json_serializable.append(None)
        with open(json_results_path, "w") as jf:
            json.dump(json_serializable, jf, indent=2, default=str, ensure_ascii=False)

        _update_job(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "brands_found": sorted(all_brands),
            "results_file": detailed_path,
            "client_format_file": client_path,
            "results_json_file": json_results_path,
        })

        # Clean up partial file on success
        _cleanup_partial(job_id)

    except Exception as e:
        _update_job(job_id, {
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e),
        })

    finally:
        with _job_lock:
            _active_jobs.pop(job_id, None)


def resume_job(job_id: str):
    """Resume a failed job from where it left off.

    Loads partial results, determines which outlets are done,
    and continues processing only the remaining ones.
    """
    job = get_job(job_id)
    if not job:
        return
    if job["status"] not in ("failed",):
        return

    # Load partial results to find already-processed serials
    partial_data = _load_partial_results(job_id)
    skip_serials = set()
    for entry in partial_data:
        serial = entry.get("serial")
        if serial is not None:
            skip_serials.add(str(serial))

    # Reset error state
    _update_job(job_id, {
        "status": "queued",
        "error_message": None,
        "completed_at": None,
    })

    thread = threading.Thread(
        target=_run_job_worker,
        args=(job_id,),
        kwargs={"skip_serials": skip_serials if skip_serials else None},
        daemon=True,
    )
    thread.start()


def delete_job(job_id: str):
    """Delete a job and its output files."""
    job = get_job(job_id)
    if job:
        for key in ["results_file", "client_format_file", "results_json_file"]:
            path = job.get(key)
            if path and Path(path).exists():
                Path(path).unlink()

    # Clean up partial results file
    _cleanup_partial(job_id)

    _db_delete_job(job_id)
