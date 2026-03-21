"""
Analytics & statistics — accuracy tracking, confusion matrix, processing stats.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

from corrections import load_corrections
from brands import BRANDS_AND_SKUS


def get_accuracy_stats(corrections_path: str = "corrections.json") -> dict:
    """
    Compute accuracy statistics from correction history.
    """
    corrections = load_corrections(corrections_path)
    real_corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    if not real_corrections:
        return {
            "total_reviewed": 0,
            "total_corrected": 0,
            "accuracy_rate": None,
            "brand_accuracy": None,
            "sku_accuracy": None,
        }

    total_reviewed = len(corrections)
    total_corrected = len(real_corrections)
    accuracy_rate = ((total_reviewed - total_corrected) / total_reviewed * 100) if total_reviewed > 0 else 0

    # Per-brand accuracy
    brand_correct = Counter()
    brand_total = Counter()
    sku_correct = Counter()
    sku_total = Counter()

    for c in corrections:
        ai_brands = set(c.get("ai_result", {}).get("brands", []))
        correct_brands = set(c.get("corrected_result", {}).get("brands", []))
        ai_skus = set(c.get("ai_result", {}).get("skus", []))
        correct_skus = set(c.get("corrected_result", {}).get("skus", []))

        # If no correction was made, AI was right for all brands
        if c.get("correction_type") == "no_change":
            for b in ai_brands:
                brand_correct[b] += 1
                brand_total[b] += 1
            for s in ai_skus:
                sku_correct[s] += 1
                sku_total[s] += 1
        else:
            # Count correctly detected brands
            for b in ai_brands | correct_brands:
                brand_total[b] += 1
                if b in ai_brands and b in correct_brands:
                    brand_correct[b] += 1

            for s in ai_skus | correct_skus:
                sku_total[s] += 1
                if s in ai_skus and s in correct_skus:
                    sku_correct[s] += 1

    # Compute per-brand accuracy
    brand_accuracy = {}
    for brand in sorted(brand_total.keys()):
        total = brand_total[brand]
        correct = brand_correct[brand]
        brand_accuracy[brand] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        }

    return {
        "total_reviewed": total_reviewed,
        "total_corrected": total_corrected,
        "accuracy_rate": round(accuracy_rate, 1),
        "brand_accuracy": brand_accuracy,
        "sku_accuracy": {
            s: {"correct": sku_correct[s], "total": sku_total[s],
                "accuracy": round(sku_correct[s] / sku_total[s] * 100, 1) if sku_total[s] > 0 else 0}
            for s in sorted(sku_total.keys())
        },
    }


def get_confusion_matrix(corrections_path: str = "corrections.json") -> dict:
    """
    Build a brand confusion matrix from corrections.
    Shows which brands get confused with each other.
    """
    corrections = load_corrections(corrections_path)
    real_corrections = [c for c in corrections if c.get("correction_type") in ("brand_swap", "brand_added", "brand_removed")]

    confusions = defaultdict(lambda: defaultdict(int))
    # confusions[false_positive_brand][should_have_been] = count

    for c in real_corrections:
        ai_brands = set(c.get("ai_result", {}).get("brands", []))
        correct_brands = set(c.get("corrected_result", {}).get("brands", []))

        # False positives: AI detected but shouldn't have
        false_positives = ai_brands - correct_brands
        # False negatives: should have been detected but wasn't
        false_negatives = correct_brands - ai_brands

        for fp in false_positives:
            for fn in false_negatives:
                confusions[fp][fn] += 1

        # Also track standalone errors
        for fp in false_positives:
            if not false_negatives:
                confusions[fp]["(should be empty)"] += 1
        for fn in false_negatives:
            if not false_positives:
                confusions["(not detected)"][fn] += 1

    # Convert to serializable format
    matrix = []
    for detected, should_be_dict in sorted(confusions.items()):
        for should_be, count in sorted(should_be_dict.items(), key=lambda x: -x[1]):
            matrix.append({
                "ai_detected": detected,
                "should_have_been": should_be,
                "count": count,
            })

    return {
        "confusions": matrix,
        "total_confusions": sum(m["count"] for m in matrix),
    }


def get_processing_stats(jobs_path: str = "jobs.json") -> dict:
    """
    Aggregate processing statistics from all completed jobs.
    """
    p = Path(jobs_path)
    if not p.exists():
        return {"total_jobs": 0}

    try:
        with open(p, "r") as f:
            jobs = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"total_jobs": 0}

    completed_jobs = [j for j in jobs if j.get("status") == "completed"]

    if not completed_jobs:
        return {
            "total_jobs": len(jobs),
            "completed_jobs": 0,
            "failed_jobs": len([j for j in jobs if j.get("status") == "failed"]),
        }

    total_outlets = sum(j.get("total_outlets", 0) for j in completed_jobs)
    total_images = sum(j.get("total_images", 0) for j in completed_jobs)
    total_errors = sum(j.get("errors", 0) for j in completed_jobs)

    # All brands found across all jobs
    all_brands = set()
    for j in completed_jobs:
        all_brands.update(j.get("brands_found", []))

    # Models used
    models_used = Counter(j.get("model", "unknown") for j in completed_jobs)

    # Processing times
    processing_times = []
    for j in completed_jobs:
        if j.get("started_at") and j.get("completed_at"):
            try:
                start = datetime.fromisoformat(j["started_at"])
                end = datetime.fromisoformat(j["completed_at"])
                processing_times.append((end - start).total_seconds())
            except (ValueError, TypeError):
                pass

    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return {
        "total_jobs": len(jobs),
        "completed_jobs": len(completed_jobs),
        "failed_jobs": len([j for j in jobs if j.get("status") == "failed"]),
        "total_outlets_processed": total_outlets,
        "total_images_processed": total_images,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_outlets * 100, 1) if total_outlets > 0 else 0,
        "unique_brands_found": sorted(all_brands),
        "models_used": dict(models_used),
        "avg_processing_time_seconds": round(avg_time, 1),
    }


def export_corrections_csv(corrections_path: str = "corrections.json") -> str:
    """
    Export corrections as CSV string for download.
    """
    corrections = load_corrections(corrections_path)

    if not corrections:
        return "No corrections to export"

    lines = [
        "ID,Timestamp,Serial,Model,Correction Type,"
        "AI Brands,Corrected Brands,AI SKUs,Corrected SKUs,Notes"
    ]

    for c in corrections:
        ai_brands = " | ".join(c.get("ai_result", {}).get("brands", []))
        correct_brands = " | ".join(c.get("corrected_result", {}).get("brands", []))
        ai_skus = " | ".join(c.get("ai_result", {}).get("skus", []))
        correct_skus = " | ".join(c.get("corrected_result", {}).get("skus", []))
        notes = c.get("notes", "").replace(",", ";").replace("\n", " ")

        lines.append(
            f"{c.get('id', '')},{c.get('timestamp', '')},"
            f"{c.get('serial', '')},{c.get('model_used', '')},"
            f"{c.get('correction_type', '')},"
            f"\"{ai_brands}\",\"{correct_brands}\","
            f"\"{ai_skus}\",\"{correct_skus}\","
            f"\"{notes}\""
        )

    return "\n".join(lines)
