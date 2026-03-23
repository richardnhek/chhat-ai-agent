"""
OCR Engine — PaddleOCR-based text extraction from cigarette pack crops.

Stage 2b of the hybrid pipeline:
  Crop → Extract all visible text → Match against known brand/SKU names
"""

import io
import re
from difflib import SequenceMatcher

from brands import BRANDS_AND_SKUS, BRAND_KHMER

# All known brand names and SKUs for fuzzy matching
_ALL_BRAND_NAMES = set(BRANDS_AND_SKUS.keys())
_ALL_SKU_NAMES = set()
for skus in BRANDS_AND_SKUS.values():
    _ALL_SKU_NAMES.update(skus)

# Common OCR mistakes for cigarette brand text
_OCR_CORRECTIONS = {
    "MEVIVS": "MEVIUS",
    "MEVLUS": "MEVIUS",
    "MEVIUUS": "MEVIUS",
    "WINSTOM": "WINSTON",
    "WINSTION": "WINSTON",
    "MARLBORE": "MARLBORO",
    "MARLBOR0": "MARLBORO",
    "COCOPALM": "COCO PALM",
    "COCOPAM": "COCO PALM",
    "COWBOY": "COW BOY",
    "COW B0Y": "COW BOY",
    "GOLDSEAL": "GOLD SEAL",
    "GOLD SEAI": "GOLD SEAL",
    "ESSE CHAGE": "ESSE CHANGE",
    "ESSE CHANGF": "ESSE CHANGE",
    "FIME": "FINE",
    "FIINE": "FINE",
    "LUXUPY": "LUXURY",
    "LUXURI": "LUXURY",
}

_ocr_reader = None


def _get_ocr():
    """Lazy-init EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except ImportError:
            return None
    return _ocr_reader


def extract_text(image_data: bytes) -> list[dict]:
    """
    Extract text from a cigarette pack crop image.

    Returns list of text detections:
    [
        {"text": "MEVIUS", "confidence": 0.95, "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]},
    ]
    """
    reader = _get_ocr()
    if reader is None:
        return _fallback_ocr(image_data)

    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)

        results = reader.readtext(img_array)

        texts = []
        for (box, text, conf) in results:
            texts.append({
                "text": text.strip(),
                "confidence": float(conf),
                "box": box,
            })

        return texts

    except Exception:
        return _fallback_ocr(image_data)


def _fallback_ocr(image_data: bytes) -> list[dict]:
    """Fallback when PaddleOCR is not available."""
    return []


def match_brand_from_text(texts: list[dict]) -> dict:
    """
    Match extracted text against known brand and SKU names.

    Returns:
    {
        "brand": "MEVIUS" or None,
        "sku": "MEVIUS ORIGINAL" or None,
        "ocr_texts": ["MEVIUS", "ORIGINAL"],
        "match_confidence": 0.92,
    }
    """
    if not texts:
        return {"brand": None, "sku": None, "ocr_texts": [], "match_confidence": 0.0}

    # Collect all text fragments
    raw_texts = [t["text"] for t in texts]
    all_text_upper = " ".join(t.upper() for t in raw_texts)

    # Apply OCR corrections
    corrected_text = all_text_upper
    for wrong, right in _OCR_CORRECTIONS.items():
        corrected_text = corrected_text.replace(wrong, right)

    # Try exact brand name match first
    best_brand = None
    best_brand_score = 0.0

    for brand in _ALL_BRAND_NAMES:
        if brand.upper() in corrected_text:
            score = len(brand) / max(len(corrected_text), 1)
            if score > best_brand_score or len(brand) > len(best_brand or ""):
                best_brand = brand
                best_brand_score = max(score, 0.9)

    # Fuzzy match if no exact match
    if best_brand is None:
        for brand in _ALL_BRAND_NAMES:
            for raw in raw_texts:
                ratio = SequenceMatcher(None, brand.upper(), raw.upper()).ratio()
                if ratio > 0.7 and ratio > best_brand_score:
                    best_brand = brand
                    best_brand_score = ratio

    # Try SKU match
    best_sku = None
    best_sku_score = 0.0

    if best_brand:
        brand_skus = BRANDS_AND_SKUS.get(best_brand, [])
        for sku in brand_skus:
            # Check if SKU-specific text is in OCR output
            sku_suffix = sku.replace(best_brand, "").strip()
            if sku_suffix and sku_suffix.upper() in corrected_text:
                best_sku = sku
                best_sku_score = 0.9
                break

        # Fuzzy SKU match
        if best_sku is None:
            for sku in brand_skus:
                sku_suffix = sku.replace(best_brand, "").strip()
                if not sku_suffix:
                    continue
                for raw in raw_texts:
                    ratio = SequenceMatcher(None, sku_suffix.upper(), raw.upper()).ratio()
                    if ratio > 0.6 and ratio > best_sku_score:
                        best_sku = sku
                        best_sku_score = ratio

        # Default to first SKU or OTHERS if brand found but no SKU match
        if best_sku is None and brand_skus:
            others = [s for s in brand_skus if "OTHERS" in s]
            best_sku = others[0] if others else brand_skus[0]
            best_sku_score = 0.5

    return {
        "brand": best_brand,
        "sku": best_sku,
        "ocr_texts": raw_texts,
        "match_confidence": best_brand_score,
        "corrected_text": corrected_text,
    }


def extract_and_match(image_data: bytes) -> dict:
    """Full OCR pipeline: extract text → match brand → match SKU."""
    texts = extract_text(image_data)
    match = match_brand_from_text(texts)
    return match
