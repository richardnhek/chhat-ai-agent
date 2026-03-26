"""
Active learning — prioritizes the most valuable images for human annotation.

Ranks unannotated images by uncertainty, brand coverage gaps, and diversity
so the team annotates images that will most improve model accuracy.
"""

import os
from collections import Counter, defaultdict
from pathlib import Path

from brands import BRANDS_AND_SKUS
from corrections import load_corrections


def _get_all_job_images() -> list[dict]:
    """Load all image records from completed jobs."""
    try:
        if os.getenv("SUPABASE_URL"):
            from supabase_db import get_all_jobs
        else:
            from database import get_all_jobs
        jobs = get_all_jobs()
    except Exception:
        return []

    images = []
    for job in jobs:
        if job.get("status") != "completed":
            continue
        results = job.get("results", [])
        if isinstance(results, str):
            import json
            try:
                results = json.loads(results)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(results, list):
            continue
        for row in results:
            urls = row.get("urls", [])
            serial = row.get("serial", "?")
            ai_brands = row.get("brands", [])
            ai_skus = row.get("skus", [])
            confidence = row.get("confidence", "medium")
            for url in urls:
                if url and isinstance(url, str):
                    images.append({
                        "url": url,
                        "serial": serial,
                        "job_id": job.get("id", ""),
                        "file_name": job.get("file_name", ""),
                        "ai_brands": ai_brands if isinstance(ai_brands, list) else [],
                        "ai_skus": ai_skus if isinstance(ai_skus, list) else [],
                        "confidence": confidence,
                    })
    return images


def _get_local_images(image_dir: str = "survey_images") -> list[dict]:
    """List local survey images."""
    p = Path(image_dir)
    if not p.exists():
        return []
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for f in sorted(p.glob(ext)):
            images.append({
                "url": "",
                "serial": f.stem.replace("serial_", "").split("_")[0],
                "file_name": f.name,
                "local_path": str(f),
                "ai_brands": [],
                "ai_skus": [],
                "confidence": "unknown",
            })
    return images


def _get_annotated_image_keys(corrections: list[dict]) -> set[str]:
    """Return a set of keys (serial or image_url) that have been annotated."""
    keys = set()
    for c in corrections:
        img_url = c.get("image_url", "")
        serial = c.get("serial", "")
        if img_url:
            keys.add(img_url)
        if serial:
            keys.add(serial)
    return keys


def _brand_annotation_counts(corrections: list[dict]) -> Counter:
    """Count how many annotations exist per brand."""
    counts: Counter = Counter()
    for c in corrections:
        corrected = c.get("corrected_result", {}).get("brands", [])
        ai = c.get("ai_result", {}).get("brands", [])
        all_brands = set(corrected) | set(ai)
        for b in all_brands:
            counts[b] += 1
    return counts


def rank_images_for_annotation(image_dir: str = "survey_images", top_k: int = 20) -> list[dict]:
    """
    Rank unannotated images by how much the model would learn from them.

    Scoring criteria:
    1. Images that haven't been annotated yet (check corrections DB)
    2. Images where the model has LOW confidence (most uncertain = most valuable)
    3. Images with brands that have few training examples
    4. Images from different outlets (diversity)

    Returns list of {"image": filename, "score": float, "reason": str}
    sorted by score descending (most valuable first).
    """
    corrections = load_corrections()
    annotated_keys = _get_annotated_image_keys(corrections)
    brand_counts = _brand_annotation_counts(corrections)

    # Gather all candidate images from local directory and jobs
    candidates = []

    # Local images
    local_imgs = _get_local_images(image_dir)
    for img in local_imgs:
        key = img.get("file_name", "") or img.get("local_path", "")
        if key not in annotated_keys and img.get("serial", "") not in annotated_keys:
            candidates.append(img)

    # Job images (remote URLs)
    job_imgs = _get_all_job_images()
    for img in job_imgs:
        url = img.get("url", "")
        serial = img.get("serial", "")
        if url not in annotated_keys and serial not in annotated_keys:
            candidates.append(img)

    if not candidates:
        return []

    # Track serials seen for diversity scoring
    serials_seen: Counter = Counter()
    scored = []

    for img in candidates:
        score = 0.0
        reasons = []

        # 1. Confidence: low confidence = high value
        conf = img.get("confidence", "unknown")
        if isinstance(conf, str):
            conf_lower = conf.lower()
            if conf_lower == "low":
                score += 40
                reasons.append("model is uncertain (low confidence)")
            elif conf_lower == "medium":
                score += 25
                reasons.append("model has medium confidence")
            elif conf_lower in ("unknown", ""):
                score += 35
                reasons.append("no confidence data available")
            else:
                score += 10
        elif isinstance(conf, (int, float)):
            # Numeric confidence: lower = more valuable
            if conf < 50:
                score += 40
                reasons.append(f"model is very uncertain ({conf}% confidence)")
            elif conf < 75:
                score += 25
                reasons.append(f"model has medium confidence ({conf}%)")
            else:
                score += 10

        # 2. Brand rarity: brands with few annotations are more valuable
        ai_brands = img.get("ai_brands", [])
        if ai_brands:
            min_brand_count = min(brand_counts.get(b, 0) for b in ai_brands) if ai_brands else 0
            if min_brand_count < 3:
                score += 35
                rare = [b for b in ai_brands if brand_counts.get(b, 0) < 3]
                reasons.append(f"contains rare brand(s): {', '.join(rare)}")
            elif min_brand_count < 10:
                score += 20
                reasons.append("contains brand(s) with limited training data")
            else:
                score += 5
        else:
            # No brands detected at all — could be missed detection
            score += 30
            reasons.append("no brands detected (possible missed detection)")

        # 3. Diversity: prefer images from different outlets/serials
        serial = img.get("serial", "unknown")
        serial_penalty = serials_seen.get(serial, 0) * 5
        diversity_score = max(0, 20 - serial_penalty)
        score += diversity_score
        serials_seen[serial] += 1
        if serial_penalty == 0:
            reasons.append("from a new outlet (adds diversity)")

        # 4. Not yet annotated bonus
        score += 5
        reasons.append("not yet annotated")

        display_name = img.get("file_name", "") or img.get("url", "unknown")
        scored.append({
            "image": display_name,
            "url": img.get("url", ""),
            "local_path": img.get("local_path", ""),
            "serial": serial,
            "ai_brands": ai_brands,
            "ai_skus": img.get("ai_skus", []),
            "confidence": img.get("confidence", "unknown"),
            "score": round(score, 1),
            "reason": "; ".join(reasons) if reasons else "unannotated image",
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_annotation_stats() -> dict:
    """
    Return stats about annotation progress:
    - total_images: total survey images available
    - annotated_images: images that have been annotated
    - unannotated_images: remaining
    - annotations_per_brand: dict of brand -> count
    - brands_needing_more: list of brands with < 5 annotations
    - team_stats: annotations per model_used (shows who annotated)
    """
    corrections = load_corrections()
    brand_counts = _brand_annotation_counts(corrections)

    # Count total available images
    local_count = len(_get_local_images())
    job_count = len(_get_all_job_images())
    total_images = max(local_count, job_count, len(corrections))

    # Annotated = corrections entries
    annotated = len(corrections)

    # Annotations per source / model
    team_stats: Counter = Counter()
    for c in corrections:
        model = c.get("model_used", "unknown")
        team_stats[model] += 1

    # Brands needing more data
    all_brands = sorted(BRANDS_AND_SKUS.keys())
    brands_needing_more = []
    for brand in all_brands:
        count = brand_counts.get(brand, 0)
        if count < 5:
            brands_needing_more.append({"brand": brand, "count": count})

    brands_needing_more.sort(key=lambda x: x["count"])

    # Annotations per brand as a clean dict
    annotations_per_brand = {brand: brand_counts.get(brand, 0) for brand in all_brands}

    return {
        "total_images": total_images,
        "annotated_images": annotated,
        "unannotated_images": max(0, total_images - annotated),
        "annotations_per_brand": annotations_per_brand,
        "brands_needing_more": brands_needing_more,
        "team_stats": dict(team_stats),
    }


def get_suggested_brands() -> list[str]:
    """
    Return list of brands that need more training data,
    sorted by urgency (fewest examples first).
    """
    corrections = load_corrections()
    brand_counts = _brand_annotation_counts(corrections)

    all_brands = sorted(BRANDS_AND_SKUS.keys())
    brand_with_counts = [(brand, brand_counts.get(brand, 0)) for brand in all_brands]
    # Sort by count ascending (fewest first)
    brand_with_counts.sort(key=lambda x: x[1])

    return [b for b, _ in brand_with_counts]
