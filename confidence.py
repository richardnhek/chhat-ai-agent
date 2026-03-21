"""
Confidence scoring — computes a proper confidence score based on multiple factors.
"""

from corrections import load_corrections


def compute_confidence(
    ai_confidence: str,
    brands_found: list[str],
    skus_found: list[str],
    unidentified_packs: int,
    num_images: int,
    brands_per_image: list[list[str]] | None = None,
) -> dict:
    """
    Compute a confidence score (0-100) and level based on multiple factors.

    Returns:
        {
            "score": 0-100,
            "level": "high" | "medium" | "low",
            "factors": { ... breakdown ... }
        }
    """
    factors = {}
    score = 100.0

    # Factor 1: AI self-reported confidence (weight: 30%)
    ai_score_map = {"high": 100, "medium": 60, "low": 30}
    ai_score = ai_score_map.get(ai_confidence, 50)
    factors["ai_confidence"] = {"value": ai_confidence, "score": ai_score}
    score -= (100 - ai_score) * 0.30

    # Factor 2: Unidentified packs penalty (weight: 25%)
    if unidentified_packs > 0:
        # Each unidentified pack reduces confidence
        unid_penalty = min(unidentified_packs * 10, 50)  # Cap at 50 points
        factors["unidentified_packs"] = {"count": unidentified_packs, "penalty": unid_penalty}
        score -= unid_penalty * 0.25
    else:
        factors["unidentified_packs"] = {"count": 0, "penalty": 0}

    # Factor 3: SKU specificity (weight: 15%)
    # If we found brands but no specific SKUs, lower confidence
    if brands_found:
        sku_ratio = len(skus_found) / len(brands_found) if brands_found else 0
        if sku_ratio >= 1.0:
            sku_score = 100
        elif sku_ratio >= 0.5:
            sku_score = 70
        else:
            sku_score = 40
        factors["sku_specificity"] = {"ratio": round(sku_ratio, 2), "score": sku_score}
        score -= (100 - sku_score) * 0.15

    # Factor 4: Multi-image consistency (weight: 20%)
    # If multiple images of the same outlet, check if they agree
    if brands_per_image and len(brands_per_image) > 1:
        # Compute overlap between image results
        sets = [set(b) for b in brands_per_image if b]
        if len(sets) >= 2:
            intersection = sets[0]
            union = sets[0]
            for s in sets[1:]:
                intersection = intersection & s
                union = union | s
            consistency = len(intersection) / len(union) if union else 1.0
            consistency_score = int(consistency * 100)
        else:
            consistency_score = 100
        factors["multi_image_consistency"] = {"score": consistency_score}
        score -= (100 - consistency_score) * 0.20
    else:
        factors["multi_image_consistency"] = {"score": 100, "note": "single image"}

    # Factor 5: Historical correction rate (weight: 10%)
    # If these brands have been frequently corrected in the past, lower confidence
    corrections = load_corrections()
    if corrections and brands_found:
        brand_set = set(brands_found)
        correction_count = 0
        for c in corrections:
            ai_brands = set(c.get("ai_result", {}).get("brands", []))
            if ai_brands & brand_set:
                correction_count += 1
        # More corrections for these brands = lower confidence
        if correction_count >= 5:
            history_score = 50
        elif correction_count >= 3:
            history_score = 70
        elif correction_count >= 1:
            history_score = 85
        else:
            history_score = 100
        factors["correction_history"] = {"related_corrections": correction_count, "score": history_score}
        score -= (100 - history_score) * 0.10
    else:
        factors["correction_history"] = {"related_corrections": 0, "score": 100}

    # Clamp score
    final_score = max(0, min(100, int(score)))

    # Determine level
    if final_score >= 80:
        level = "high"
    elif final_score >= 55:
        level = "medium"
    else:
        level = "low"

    return {
        "score": final_score,
        "level": level,
        "factors": factors,
    }
