"""
Hybrid Pipeline — The full Option 3 detect → crop → recognize pipeline.

Architecture:
  1. Image QA (blur check, reject garbage)
  2. RF-DETR Detection (find all packs, return crops)
  3. For each crop:
     a. Signal 1: PaddleOCR (read brand/SKU text)
     b. Signal 2: Embedding search (visual similarity against 238 reference images)
     c. If OCR + embedding agree → high confidence, accept
     d. If OCR confident alone → accept
     e. If uncertain or disagreement → send crop to Gemini arbitrator
  4. Fusion (combine all signals, resolve conflicts)
  5. Confidence scoring
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from brands import BRANDS_AND_SKUS
from detector import detect_packs, crop_image_region
from ocr_engine import extract_and_match
from enhancements import enhance_image, compute_blur_score
from confidence import compute_confidence
from embedding_search import find_matching_sku
from rate_limiter import RateLimiter
from logger import get_logger

logger = get_logger()

# Threshold: if OCR match confidence is below this, escalate to Gemini
OCR_CONFIDENCE_THRESHOLD = 0.75

# Threshold: embedding similarity above this is considered a strong visual match
EMBEDDING_CONFIDENCE_THRESHOLD = 0.85

# Threshold: if blur score is below this, flag as low quality
BLUR_THRESHOLD = 50.0


def _classify_crop_with_gemini(
    crop_data: bytes,
    media_type: str,
    model: str,
    api_keys: dict,
    known_brands_in_image: list[str] | None = None,
) -> dict:
    """Send a single crop to Gemini for brand/SKU identification."""
    from image_analyzer import analyze_image, ANALYSIS_PROMPT

    # Build a focused prompt for a single crop
    prompt = f"""This is a close-up crop of a single cigarette pack from a store display case.
Identify the EXACT brand and SKU of this cigarette pack.

Look at:
- Brand name text on the pack
- Pack color (red=RED/HARD PACK, blue=LIGHTS/MENTHOL, gold=GOLD, black=ORIGINAL, green=MENTHOL)
- Logo design
- Pack shape (slim=SLIMS, regular=HARD PACK/FF)

Official brand list:
{chr(10).join(f'- {b}: [{", ".join(skus)}]' for b, skus in BRANDS_AND_SKUS.items())}

Respond in JSON (no markdown):
{{"brand": "<brand name or null>", "sku": "<exact SKU or null>", "confidence": "<high|medium|low>", "notes": "<what you see>"}}
"""

    result = analyze_image(
        crop_data, media_type,
        model=model, api_keys=api_keys,
        correction_context="",
    )

    # Parse the result — it might come back in brands_found format
    if "brands_found" in result and result["brands_found"]:
        entry = result["brands_found"][0]
        return {
            "brand": entry.get("brand"),
            "sku": entry.get("skus", [None])[0] if entry.get("skus") else None,
            "confidence": result.get("confidence", "medium"),
            "notes": entry.get("notes", ""),
            "source": "gemini_arbitrator",
        }
    elif "brand" in result:
        return {
            "brand": result.get("brand"),
            "sku": result.get("sku"),
            "confidence": result.get("confidence", "medium"),
            "notes": result.get("notes", ""),
            "source": "gemini_arbitrator",
        }

    return {"brand": None, "sku": None, "confidence": "low", "notes": "Gemini could not identify", "source": "gemini_arbitrator"}


def _process_single_crop(
    crop_data: bytes,
    media_type: str,
    model: str,
    api_keys: dict,
    limiter: RateLimiter | None = None,
) -> dict:
    """
    Process a single crop through OCR + Embedding Search → optional Gemini arbitrator.

    Signal fusion logic:
      - OCR confident + embedding agrees → high confidence, skip Gemini
      - OCR confident alone → accept (medium-high confidence)
      - Embedding confident + OCR weak → accept embedding (medium confidence)
      - Both weak or disagree → escalate to Gemini
    """
    # ── Signal 1: OCR ────────────────────────────────────────────────
    ocr_result = extract_and_match(crop_data)
    ocr_brand = ocr_result.get("brand")
    ocr_sku = ocr_result.get("sku")
    ocr_conf = ocr_result.get("match_confidence", 0)

    # ── Signal 2: Embedding search ───────────────────────────────────
    try:
        emb_matches = find_matching_sku(crop_data, top_k=3)
        emb_top = emb_matches[0] if emb_matches else None
        emb_brand = emb_top["brand"] if emb_top else None
        emb_sku = emb_top["sku"] if emb_top else None
        emb_sim = emb_top["similarity"] if emb_top else 0
        emb_ref = emb_top["reference_image"] if emb_top else ""
    except Exception as e:
        logger.warning(f"Embedding search failed: {e}")
        emb_top = None
        emb_brand = None
        emb_sku = None
        emb_sim = 0
        emb_ref = ""

    # ── Fusion logic ─────────────────────────────────────────────────

    # Case 1: OCR confident + embedding agrees → high confidence
    if (ocr_brand and ocr_conf >= OCR_CONFIDENCE_THRESHOLD
            and emb_brand and emb_sim >= EMBEDDING_CONFIDENCE_THRESHOLD
            and ocr_brand == emb_brand):
        # Both signals agree — very high confidence, skip Gemini
        chosen_sku = ocr_sku or emb_sku
        return {
            "brand": ocr_brand,
            "sku": chosen_sku,
            "confidence": "high",
            "notes": (f"OCR+Embedding agree. OCR: {', '.join(ocr_result['ocr_texts'][:3])}. "
                      f"Embedding: {emb_ref} (sim={emb_sim:.3f})"),
            "source": "ocr+embedding",
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }

    # Case 2: OCR confident alone → accept
    if ocr_brand and ocr_conf >= OCR_CONFIDENCE_THRESHOLD:
        return {
            "brand": ocr_brand,
            "sku": ocr_sku,
            "confidence": "high" if ocr_conf > 0.85 else "medium",
            "notes": (f"OCR detected: {', '.join(ocr_result['ocr_texts'][:5])}. "
                      f"Embedding top: {emb_brand} (sim={emb_sim:.3f})"),
            "source": "ocr",
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }

    # Case 3: Embedding confident + OCR weak → use embedding
    if emb_brand and emb_sim >= EMBEDDING_CONFIDENCE_THRESHOLD and not ocr_brand:
        return {
            "brand": emb_brand,
            "sku": emb_sku,
            "confidence": "medium",
            "notes": (f"Embedding match: {emb_ref} (sim={emb_sim:.3f}). "
                      f"OCR had no match."),
            "source": "embedding",
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }

    # Case 4: Both have results but disagree, or both weak → escalate to Gemini
    if limiter:
        limiter.wait()

    gemini_result = _classify_crop_with_gemini(crop_data, media_type, model, api_keys)

    # Build disagreement context for notes
    signal_notes = []
    if ocr_brand:
        signal_notes.append(f"OCR={ocr_brand}(conf={ocr_conf:.2f})")
    if emb_brand:
        signal_notes.append(f"Embedding={emb_brand}(sim={emb_sim:.3f})")
    signal_summary = ", ".join(signal_notes) if signal_notes else "no pre-signals"

    # Fuse all three signals
    if gemini_result["brand"]:
        # Check how many signals agree with Gemini
        agreements = []
        if ocr_brand == gemini_result["brand"]:
            agreements.append("OCR")
        if emb_brand == gemini_result["brand"]:
            agreements.append("Embedding")

        if len(agreements) >= 2:
            confidence = "high"
            source = "ocr+embedding+gemini"
        elif len(agreements) == 1:
            confidence = "high" if agreements[0] == "Embedding" and emb_sim > 0.8 else "medium"
            source = f"{agreements[0].lower()}+gemini"
        else:
            confidence = "medium"
            source = "gemini_arbitrator"

        return {
            "brand": gemini_result["brand"],
            "sku": gemini_result["sku"] or ocr_sku or emb_sku,
            "confidence": confidence,
            "notes": f"Gemini arbitrated. Pre-signals: {signal_summary}. Agrees with: {agreements or 'neither'}.",
            "source": source,
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }

    # Gemini also failed — use best available signal
    if ocr_brand and ocr_conf > (emb_sim if emb_brand else 0):
        return {
            "brand": ocr_brand,
            "sku": ocr_sku,
            "confidence": "low",
            "notes": f"Only OCR detected (low confidence): {ocr_result['ocr_texts'][:3]}",
            "source": "ocr_low",
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }
    elif emb_brand and emb_sim > 0.6:
        return {
            "brand": emb_brand,
            "sku": emb_sku,
            "confidence": "low",
            "notes": f"Only embedding match (weak): {emb_ref} (sim={emb_sim:.3f})",
            "source": "embedding_low",
            "ocr_confidence": ocr_conf,
            "embedding_similarity": emb_sim,
        }

    return {
        "brand": None,
        "sku": None,
        "confidence": "low",
        "notes": "No signal could identify this pack (OCR, Embedding, Gemini all failed)",
        "source": "unidentified",
        "ocr_confidence": ocr_conf,
        "embedding_similarity": emb_sim,
    }


def analyze_hybrid(
    image_data: bytes,
    media_type: str = "image/jpeg",
    model: str = "gemini-2.5-pro",
    api_keys: dict | None = None,
    correction_context: str = "",
    rfdetr_model_path: str | None = None,
) -> dict:
    """
    Full hybrid pipeline: detect → crop → OCR → arbitrate → fuse.

    This is the Option 3 enterprise pipeline.
    """
    api_keys = api_keys or {}
    start_time = time.time()

    # ── Step 1: Image QA ─────────────────────────────────────────────
    blur_score = compute_blur_score(image_data)
    if blur_score < BLUR_THRESHOLD:
        logger.warning(f"Low quality image (blur={blur_score:.1f}), enhancing...")

    # Enhance image for better detection
    enhanced_data = enhance_image(image_data)

    # ── Step 2: Detect packs ─────────────────────────────────────────
    detections = detect_packs(enhanced_data, confidence_threshold=0.3, model_path=rfdetr_model_path)

    is_fallback = any(d.get("is_fallback") for d in detections)

    if is_fallback:
        # RF-DETR not available — fall back to full-image Gemini analysis
        logger.info("RF-DETR not available, using full-image Gemini analysis")
        from enhancements import analyze_image_enhanced

        result = analyze_image_enhanced(
            image_data, media_type,
            model=model, api_keys=api_keys,
            correction_context=correction_context,
            enable_enhancement=True,
            enable_ocr=True,
            enable_sku_refinement=True,
        )
        result["pipeline"] = "fallback_gemini"
        result["blur_score"] = blur_score
        return result

    # ── Step 3: Process each crop ────────────────────────────────────
    limiter = RateLimiter(model)
    all_brands = set()
    all_skus = set()
    crop_results = []
    gemini_calls = 0
    ocr_only = 0
    embedding_resolved = 0
    ocr_embedding_agree = 0
    unidentified = 0

    for det in detections:
        crop_data = det["crop"]

        crop_result = _process_single_crop(
            crop_data, "image/jpeg", model, api_keys, limiter
        )

        crop_result["box"] = det["box"]
        crop_result["detection_confidence"] = det["confidence"]
        crop_results.append(crop_result)

        if crop_result["brand"] and crop_result["brand"] in BRANDS_AND_SKUS:
            all_brands.add(crop_result["brand"])
            if crop_result["sku"]:
                all_skus.add(crop_result["sku"])

        source = crop_result["source"]
        if source in ("gemini_arbitrator", "gemini_override", "ocr+gemini",
                       "ocr+embedding+gemini", "ocr+gemini", "embedding+gemini"):
            gemini_calls += 1
        if source == "ocr":
            ocr_only += 1
        elif source == "ocr+embedding":
            ocr_embedding_agree += 1
        elif source == "embedding":
            embedding_resolved += 1
        elif source == "unidentified":
            unidentified += 1

    # ── Step 4: Build result ─────────────────────────────────────────
    brands_sorted = sorted(all_brands)
    skus_sorted = sorted(all_skus)

    # Determine overall confidence
    conf_values = {"high": 3, "medium": 2, "low": 1}
    if crop_results:
        avg_conf = sum(conf_values.get(r.get("confidence", "low"), 1) for r in crop_results) / len(crop_results)
        if avg_conf >= 2.5:
            overall_confidence = "high"
        elif avg_conf >= 1.5:
            overall_confidence = "medium"
        else:
            overall_confidence = "low"
    else:
        overall_confidence = "high"  # No packs detected = confident there are none

    elapsed = time.time() - start_time

    brands_found = []
    for brand in brands_sorted:
        brand_skus = [s for s in skus_sorted if s in BRANDS_AND_SKUS.get(brand, [])]
        brand_crops = [r for r in crop_results if r.get("brand") == brand]
        notes_parts = [r.get("notes", "") for r in brand_crops if r.get("notes")]
        brands_found.append({
            "brand": brand,
            "skus": brand_skus if brand_skus else [brand],
            "notes": "; ".join(notes_parts[:2]),
            "pack_count": len(brand_crops),
        })

    return {
        "brands_found": brands_found,
        "brand_count": len(brands_sorted),
        "unidentified_packs": unidentified,
        "confidence": overall_confidence,
        "notes": (
            f"Hybrid pipeline: {len(detections)} packs detected, "
            f"{ocr_only} resolved by OCR, {ocr_embedding_agree} by OCR+Embedding, "
            f"{embedding_resolved} by Embedding alone, {gemini_calls} escalated to Gemini, "
            f"{unidentified} unidentified. Processed in {elapsed:.1f}s."
        ),
        "pipeline": "hybrid",
        "blur_score": blur_score,
        "stats": {
            "total_packs": len(detections),
            "ocr_resolved": ocr_only,
            "ocr_embedding_agree": ocr_embedding_agree,
            "embedding_resolved": embedding_resolved,
            "gemini_escalated": gemini_calls,
            "unidentified": unidentified,
            "processing_time": round(elapsed, 1),
        },
    }
