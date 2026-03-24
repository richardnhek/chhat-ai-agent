"""
Option 2 Pipeline — Fully self-hosted, zero API calls in production.

Architecture:
  1. Image Enhancement (OpenCV — local)
  2. RF-DETR Detection (local GPU/CPU)
  3. For each crop:
     a. EasyOCR text extraction (local)
     b. Embedding search against reference catalog (local, ResNet50)
     c. Fusion: OCR + Embedding → brand/SKU
     d. Low confidence → flag for human review (NOT Gemini)
  4. Confidence scoring
  5. Output

No API calls. No internet required. Runs entirely offline.
"""

import io
import time
from PIL import Image

from brands import BRANDS_AND_SKUS
from detector import detect_packs
from ocr_engine import extract_and_match
from embedding_search import find_matching_sku
from enhancements import enhance_image, compute_blur_score
from confidence import compute_confidence
from logger import get_logger

logger = get_logger()

OCR_CONFIDENCE_THRESHOLD = 0.70
EMBEDDING_CONFIDENCE_THRESHOLD = 0.80
BLUR_THRESHOLD = 50.0


def _classify_crop_local(crop_data: bytes) -> dict:
    """
    Classify a single crop using ONLY local models (no API calls).
    Signal 1: EasyOCR → fuzzy match against brand/SKU list
    Signal 2: Embedding similarity search against 233 reference images
    """
    # Signal 1: OCR
    ocr_result = extract_and_match(crop_data)
    ocr_brand = ocr_result.get("brand")
    ocr_sku = ocr_result.get("sku")
    ocr_conf = ocr_result.get("match_confidence", 0)

    # Signal 2: Embedding search
    try:
        emb_matches = find_matching_sku(crop_data, top_k=3)
        emb_top = emb_matches[0] if emb_matches else None
        emb_brand = emb_top["brand"] if emb_top else None
        emb_sku = emb_top["sku"] if emb_top else None
        emb_sim = emb_top["similarity"] if emb_top else 0
    except Exception:
        emb_brand = None
        emb_sku = None
        emb_sim = 0

    # Fusion logic — no Gemini fallback, just local signals

    # Case 1: Both agree → high confidence
    if (ocr_brand and ocr_conf >= OCR_CONFIDENCE_THRESHOLD
            and emb_brand and emb_sim >= EMBEDDING_CONFIDENCE_THRESHOLD
            and ocr_brand == emb_brand):
        return {
            "brand": ocr_brand,
            "sku": ocr_sku or emb_sku,
            "confidence": "high",
            "source": "ocr+embedding",
            "notes": f"OCR+Embedding agree (OCR:{ocr_conf:.2f}, Emb:{emb_sim:.3f})",
        }

    # Case 2: OCR confident → trust it
    if ocr_brand and ocr_conf >= OCR_CONFIDENCE_THRESHOLD:
        conf = "high" if ocr_conf > 0.85 else "medium"
        return {
            "brand": ocr_brand,
            "sku": ocr_sku,
            "confidence": conf,
            "source": "ocr",
            "notes": f"OCR detected (conf:{ocr_conf:.2f}), Embedding: {emb_brand}({emb_sim:.3f})",
        }

    # Case 3: Embedding confident → trust it
    if emb_brand and emb_sim >= EMBEDDING_CONFIDENCE_THRESHOLD:
        return {
            "brand": emb_brand,
            "sku": emb_sku,
            "confidence": "medium",
            "source": "embedding",
            "notes": f"Embedding match (sim:{emb_sim:.3f}), OCR weak: {ocr_brand}({ocr_conf:.2f})",
        }

    # Case 4: Weaker signals — use best available
    if ocr_brand and ocr_conf > 0.5:
        return {
            "brand": ocr_brand,
            "sku": ocr_sku,
            "confidence": "low",
            "source": "ocr_low",
            "notes": f"Weak OCR only (conf:{ocr_conf:.2f}). Needs human review.",
        }

    if emb_brand and emb_sim > 0.6:
        return {
            "brand": emb_brand,
            "sku": emb_sku,
            "confidence": "low",
            "source": "embedding_low",
            "notes": f"Weak embedding only (sim:{emb_sim:.3f}). Needs human review.",
        }

    # Case 5: Nothing found → flag for review
    return {
        "brand": None,
        "sku": None,
        "confidence": "low",
        "source": "unidentified",
        "notes": "Neither OCR nor embedding could identify. Flagged for human review.",
    }


def analyze_v2(
    image_data: bytes,
    media_type: str = "image/jpeg",
    rfdetr_model_path: str | None = None,
    correction_context: str = "",
) -> dict:
    """
    Option 2 pipeline: fully self-hosted analysis, zero API calls.
    """
    start_time = time.time()

    # Step 1: Image QA + Enhancement
    blur_score = compute_blur_score(image_data)
    enhanced_data = enhance_image(image_data)

    # Step 2: Detect packs
    detections = detect_packs(enhanced_data, confidence_threshold=0.3, model_path=rfdetr_model_path)
    is_fallback = any(d.get("is_fallback") for d in detections)

    if is_fallback:
        # RF-DETR not fine-tuned yet — run OCR + embedding on full image
        logger.info("RF-DETR not fine-tuned, running full-image local analysis")
        crop_result = _classify_crop_local(enhanced_data)

        brands_found = []
        if crop_result["brand"] and crop_result["brand"] in BRANDS_AND_SKUS:
            brands_found.append({
                "brand": crop_result["brand"],
                "skus": [crop_result["sku"]] if crop_result["sku"] else [],
                "notes": crop_result["notes"],
            })

        return {
            "brands_found": brands_found,
            "brand_count": len(brands_found),
            "unidentified_packs": 1 if not brands_found else 0,
            "confidence": crop_result["confidence"],
            "notes": f"Fallback mode (RF-DETR not trained). {crop_result['notes']}",
            "pipeline": "v2_fallback",
            "blur_score": blur_score,
            "stats": {"total_packs": 0, "processing_time": round(time.time() - start_time, 1)},
        }

    # Step 3: Process each crop locally
    all_brands = set()
    all_skus = set()
    crop_results = []
    ocr_resolved = 0
    embedding_resolved = 0
    both_agree = 0
    unidentified = 0

    for det in detections:
        crop_data = det["crop"]
        result = _classify_crop_local(crop_data)
        result["box"] = det["box"]
        result["detection_confidence"] = det["confidence"]
        crop_results.append(result)

        if result["brand"] and result["brand"] in BRANDS_AND_SKUS:
            all_brands.add(result["brand"])
            if result["sku"]:
                all_skus.add(result["sku"])

        source = result["source"]
        if source == "ocr+embedding":
            both_agree += 1
        elif source == "ocr":
            ocr_resolved += 1
        elif source == "embedding":
            embedding_resolved += 1
        elif source == "unidentified":
            unidentified += 1

    # Step 4: Build result
    brands_sorted = sorted(all_brands)
    skus_sorted = sorted(all_skus)

    conf_values = {"high": 3, "medium": 2, "low": 1}
    if crop_results:
        avg_conf = sum(conf_values.get(r.get("confidence", "low"), 1) for r in crop_results) / len(crop_results)
        overall_confidence = "high" if avg_conf >= 2.5 else ("medium" if avg_conf >= 1.5 else "low")
    else:
        overall_confidence = "high"

    elapsed = time.time() - start_time

    brands_found = []
    for brand in brands_sorted:
        brand_skus = [s for s in skus_sorted if s in BRANDS_AND_SKUS.get(brand, [])]
        brand_crops = [r for r in crop_results if r.get("brand") == brand]
        notes = "; ".join(r.get("notes", "") for r in brand_crops[:2])
        brands_found.append({
            "brand": brand,
            "skus": brand_skus if brand_skus else [brand],
            "notes": notes,
            "pack_count": len(brand_crops),
        })

    return {
        "brands_found": brands_found,
        "brand_count": len(brands_sorted),
        "unidentified_packs": unidentified,
        "confidence": overall_confidence,
        "notes": (
            f"Self-hosted pipeline (zero API calls): {len(detections)} packs detected, "
            f"{both_agree} OCR+Embedding agree, {ocr_resolved} OCR only, "
            f"{embedding_resolved} Embedding only, {unidentified} unidentified. "
            f"Processed in {elapsed:.1f}s."
        ),
        "pipeline": "v2_self_hosted",
        "blur_score": blur_score,
        "stats": {
            "total_packs": len(detections),
            "ocr_embedding_agree": both_agree,
            "ocr_resolved": ocr_resolved,
            "embedding_resolved": embedding_resolved,
            "unidentified": unidentified,
            "processing_time": round(elapsed, 1),
            "api_calls": 0,
        },
    }
