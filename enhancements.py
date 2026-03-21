"""
Image analysis enhancements:
1. Super-resolution / image sharpening for blurry images
2. OCR text extraction to assist brand/SKU detection
3. Two-pass SKU refinement (brands first, then focused SKU identification)
4. Cross-model validation (run through 2 models, compare)
5. Reference image context from brand book
"""

import io
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageFilter

from brands import BRANDS_AND_SKUS, get_brand_list_for_prompt


# ═══════════════════════════════════════════════════════════════════════════
# 1. IMAGE ENHANCEMENT (Super-resolution / sharpening / deglare)
# ═══════════════════════════════════════════════════════════════════════════

def enhance_image(image_data: bytes) -> bytes:
    """
    Enhance image quality for better brand detection.
    Applies: denoising, sharpening, contrast enhancement, deglare.
    """
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_data

    # Step 1: Denoise (removes grain/noise without losing edges)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    # Step 2: CLAHE contrast enhancement (adaptive, works per-region)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Step 3: Unsharp mask sharpening (makes text and edges crisp)
    gaussian = cv2.GaussianBlur(contrast_enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.5, gaussian, -0.5, 0)

    # Step 4: Mild dehazing (reduces glass glare effect)
    # Simple approach: reduce brightness of very bright pixels (glare spots)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Tone down extremely bright areas (likely glare)
    v_clipped = np.clip(v.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
    dehazed = cv2.cvtColor(cv2.merge([h, s, v_clipped]), cv2.COLOR_HSV2BGR)

    # Encode back to JPEG
    _, buffer = cv2.imencode('.jpg', dehazed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def compute_blur_score(image_data: bytes) -> float:
    """
    Compute a blur score for the image (higher = sharper).
    Uses the variance of the Laplacian.
    Returns a score where < 50 is very blurry, 50-100 is moderate, > 100 is sharp.
    """
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


# ═══════════════════════════════════════════════════════════════════════════
# 2. OCR TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

OCR_PROMPT = """Look at this image of a store display case. Extract ALL visible text you can read on cigarette packs and boxes.

Include:
- Brand names (even partial, upside-down, or sideways text)
- SKU variant text (e.g., "MENTHOL", "LIGHTS", "GOLD", "CHANGE", "ORIGINAL", "RED", "FF")
- Any numbers or codes visible on packs

Return ONLY a JSON list of strings — every distinct text snippet you can read:
["text1", "text2", "text3"]

If no text is readable, return: []
Do NOT include text from non-cigarette products.
"""


def extract_text_ocr(image_data: bytes, media_type: str, model: str, api_keys: dict) -> list[str]:
    """
    Use a lightweight AI call to extract all readable text from the image.
    This text is then used to assist brand/SKU detection.
    """
    from image_analyzer import _parse_response, MODEL_REGISTRY, _PROVIDER_FNS
    import os

    # Use a cheap/fast model for OCR — prefer Flash variants
    ocr_model = model
    if "pro" in model:
        # Try to use a flash variant for OCR (cheaper)
        if "gemini" in model:
            ocr_model = "gemini-2.5-flash"
        elif "claude" in model:
            ocr_model = "claude-haiku-4-5"

    if ocr_model in MODEL_REGISTRY:
        provider, model_id = MODEL_REGISTRY[ocr_model]
    else:
        provider = "gemini"
        model_id = ocr_model

    api_key = api_keys.get(provider, "") or os.getenv({
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
    }.get(provider, ""), "")

    if not api_key:
        return []

    fn = _PROVIDER_FNS.get(provider)
    if not fn:
        return []

    try:
        result = fn(image_data, media_type, model_id, api_key, prompt=OCR_PROMPT)
        # Parse the text list
        if isinstance(result, dict) and "brands_found" in result:
            # The parser returned our standard format — means it couldn't parse the list
            return []

        # Try to extract a JSON list from the response
        text = result if isinstance(result, str) else json.dumps(result)
        match = re.search(r'\[[\s\S]*?\]', text)
        if match:
            texts = json.loads(match.group())
            if isinstance(texts, list):
                return [str(t).strip() for t in texts if t]
        return []
    except Exception:
        return []


def format_ocr_context(texts: list[str]) -> str:
    """Format OCR results as additional context for the main analysis prompt."""
    if not texts:
        return ""
    text_list = ", ".join(f'"{t}"' for t in texts[:30])  # Cap at 30 items
    return (
        f"\n\nOCR PRE-SCAN RESULTS:\n"
        f"Text detected on packs in this image: [{text_list}]\n"
        f"Use this text to help identify brands and SKUs more accurately. "
        f"Match detected text against the official brand/SKU list."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TWO-PASS SKU REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════

def build_sku_refinement_prompt(brands_found: list[str]) -> str:
    """Build a focused prompt for SKU refinement given known brands."""
    brand_sku_details = []
    for brand in brands_found:
        skus = BRANDS_AND_SKUS.get(brand, [])
        if skus:
            brand_sku_details.append(f"- {brand}: possible SKUs = [{', '.join(skus)}]")

    return f"""You already know this image contains these cigarette brands: {', '.join(brands_found)}

For EACH brand, determine the EXACT SKU variant. Look carefully at:
- Pack COLOR (red=RED/HARD PACK/FF, blue=LIGHTS/BLUE/MENTHOL, gold=GOLD, green=MENTHOL, black=ORIGINAL)
- Text on the pack (MENTHOL, LIGHTS, CHANGE, ORIGINAL, GOLD, etc.)
- Pack SIZE (slim packs = SLIMS/SUPER SLIMS, compact = COMPACT)
- If you cannot determine the exact SKU, use OTHERS

Here are the valid SKUs for each detected brand:
{chr(10).join(brand_sku_details)}

Respond in this EXACT JSON format (no markdown):
{{
    "sku_assignments": [
        {{"brand": "<brand>", "sku": "<EXACT SKU from list>", "reasoning": "<what visual clue led to this SKU>"}}
    ]
}}
"""


def refine_skus(
    image_data: bytes,
    media_type: str,
    brands_found: list[str],
    model: str,
    api_keys: dict,
) -> list[dict]:
    """
    Second pass: given the brands already detected, do a focused analysis
    to identify the specific SKU for each brand.
    """
    if not brands_found:
        return []

    from image_analyzer import MODEL_REGISTRY, _PROVIDER_FNS, _parse_response
    import os

    prompt = build_sku_refinement_prompt(brands_found)

    if model in MODEL_REGISTRY:
        provider, model_id = MODEL_REGISTRY[model]
    else:
        provider = "gemini"
        model_id = model

    api_key = api_keys.get(provider, "") or os.getenv({
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
    }.get(provider, ""), "")

    if not api_key:
        return []

    fn = _PROVIDER_FNS.get(provider)
    if not fn:
        return []

    try:
        result = fn(image_data, media_type, model_id, api_key, prompt=prompt)

        if isinstance(result, dict):
            assignments = result.get("sku_assignments", [])
            # Validate SKUs against official list
            validated = []
            for a in assignments:
                brand = a.get("brand", "")
                sku = a.get("sku", "")
                valid_skus = BRANDS_AND_SKUS.get(brand, [])
                if sku in valid_skus:
                    validated.append(a)
                elif valid_skus:
                    # Check for partial match
                    matched = False
                    for vs in valid_skus:
                        if sku.upper() in vs.upper() or vs.upper() in sku.upper():
                            validated.append({"brand": brand, "sku": vs, "reasoning": a.get("reasoning", "")})
                            matched = True
                            break
                    if not matched:
                        # Fall back to OTHERS if available
                        others = [s for s in valid_skus if "OTHERS" in s]
                        if others:
                            validated.append({"brand": brand, "sku": others[0], "reasoning": "Could not determine specific SKU"})
                        elif valid_skus:
                            validated.append({"brand": brand, "sku": valid_skus[0], "reasoning": "Defaulted to first SKU"})
            return validated
        return []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 4. CROSS-MODEL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def cross_validate(
    image_data: bytes,
    media_type: str,
    models: list[str],
    api_keys: dict,
    correction_context: str = "",
) -> dict:
    """
    Run the same image through multiple models and combine results.
    Returns a validated result with agreement-based confidence.
    """
    from image_analyzer import analyze_image

    results = []

    # Run models in parallel
    def _run_model(m):
        return analyze_image(image_data, media_type, model=m, api_keys=api_keys, correction_context=correction_context)

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(_run_model, m): m for m in models}
        for future in futures:
            try:
                result = future.result()
                if "error" not in result:
                    results.append({"model": futures[future], "result": result})
            except Exception:
                pass

    if not results:
        return {"error": "All models failed"}

    if len(results) == 1:
        return results[0]["result"]

    # Aggregate: count brand votes across models
    brand_votes: dict[str, int] = {}
    sku_votes: dict[str, int] = {}
    all_brands_found = []

    for r in results:
        brands_in_result = set()
        for entry in r["result"].get("brands_found", []):
            brand = entry.get("brand", "")
            brand_votes[brand] = brand_votes.get(brand, 0) + 1
            brands_in_result.add(brand)
            for sku in entry.get("skus", []):
                sku_votes[sku] = sku_votes.get(sku, 0) + 1
        all_brands_found.append(brands_in_result)

    num_models = len(results)
    threshold = num_models / 2  # Majority vote

    # Brands with majority agreement
    agreed_brands = [b for b, count in brand_votes.items() if count > threshold]
    agreed_skus = [s for s, count in sku_votes.items() if count > threshold]

    # Disputed brands (only detected by minority)
    disputed_brands = [b for b, count in brand_votes.items() if 0 < count <= threshold]

    # Compute agreement-based confidence
    if all_brands_found:
        total_unique = len(set().union(*all_brands_found))
        agreed_count = len(agreed_brands)
        agreement_ratio = agreed_count / total_unique if total_unique > 0 else 1.0
    else:
        agreement_ratio = 1.0

    # Build combined result
    brands_found = []
    for brand in agreed_brands:
        brand_skus = [s for s in agreed_skus if s.startswith(brand)]
        if not brand_skus:
            # Take SKU from any model that detected this brand
            for r in results:
                for entry in r["result"].get("brands_found", []):
                    if entry.get("brand") == brand:
                        brand_skus = entry.get("skus", [])
                        break
                if brand_skus:
                    break
        brands_found.append({
            "brand": brand,
            "skus": brand_skus,
            "notes": f"Agreed by {brand_votes[brand]}/{num_models} models",
        })

    # Determine confidence
    unidentified = max(r["result"].get("unidentified_packs", 0) for r in results)

    if agreement_ratio >= 0.9:
        confidence = "high"
    elif agreement_ratio >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    notes = f"Cross-validated with {num_models} models. Agreement: {agreement_ratio:.0%}."
    if disputed_brands:
        notes += f" Disputed brands (no majority): {', '.join(disputed_brands)}."

    return {
        "brands_found": brands_found,
        "brand_count": len(agreed_brands),
        "unidentified_packs": unidentified,
        "confidence": confidence,
        "notes": notes,
        "validation": {
            "models_used": [r["model"] for r in results],
            "agreement_ratio": round(agreement_ratio, 2),
            "disputed_brands": disputed_brands,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENHANCED ANALYSIS PIPELINE (combines all enhancements)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_image_enhanced(
    image_data: bytes,
    media_type: str,
    model: str = "gemini-2.5-pro",
    api_keys: dict | None = None,
    correction_context: str = "",
    enable_enhancement: bool = True,
    enable_ocr: bool = True,
    enable_sku_refinement: bool = True,
    enable_cross_validation: bool = False,
    cross_validation_models: list[str] | None = None,
) -> dict:
    """
    Enhanced analysis pipeline that chains multiple improvements:
    1. Image enhancement (sharpening, deglare, contrast)
    2. OCR text pre-extraction
    3. Main brand detection (with OCR context + correction context)
    4. Two-pass SKU refinement
    5. Optional cross-model validation
    """
    from image_analyzer import analyze_image

    api_keys = api_keys or {}
    enhanced_data = image_data

    # Step 1: Image enhancement
    blur_score = compute_blur_score(image_data)
    if enable_enhancement:
        enhanced_data = enhance_image(image_data)

    # Step 2: OCR text extraction (parallel with enhancement)
    ocr_context = ""
    if enable_ocr:
        ocr_texts = extract_text_ocr(enhanced_data, media_type, model, api_keys)
        ocr_context = format_ocr_context(ocr_texts)

    # Step 3: Main analysis (or cross-validation)
    full_context = correction_context
    if ocr_context:
        full_context = (full_context + "\n" + ocr_context) if full_context else ocr_context

    if enable_cross_validation and cross_validation_models:
        result = cross_validate(
            enhanced_data, media_type,
            models=cross_validation_models,
            api_keys=api_keys,
            correction_context=full_context,
        )
    else:
        result = analyze_image(
            enhanced_data, media_type,
            model=model, api_keys=api_keys,
            correction_context=full_context,
        )

    if "error" in result:
        return result

    # Step 4: Two-pass SKU refinement
    if enable_sku_refinement:
        brands_detected = [e.get("brand", "") for e in result.get("brands_found", [])]
        brands_detected = [b for b in brands_detected if b in BRANDS_AND_SKUS]

        if brands_detected:
            refined = refine_skus(enhanced_data, media_type, brands_detected, model, api_keys)
            if refined:
                # Merge refined SKUs back into the result
                sku_map = {r["brand"]: r["sku"] for r in refined}
                for entry in result.get("brands_found", []):
                    brand = entry.get("brand", "")
                    if brand in sku_map:
                        entry["skus"] = [sku_map[brand]]
                        entry["notes"] = entry.get("notes", "") + f" (SKU refined: {sku_map[brand]})"

    # Add blur score to result
    result["blur_score"] = round(blur_score, 1)
    result["image_enhanced"] = enable_enhancement

    return result
