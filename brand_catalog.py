"""
Brand Catalog loader — checks for brand_catalog.json overrides,
falls back to hardcoded brands.py data.

The JSON file is created/updated by the Brand Catalog UI page.
"""

import json
from pathlib import Path
from brands import BRANDS_AND_SKUS, BRAND_KHMER

CATALOG_PATH = Path(__file__).parent / "brand_catalog.json"


def load_brand_catalog() -> dict:
    """
    Load the brand catalog. Returns a dict with keys:
    - brands: {brand_name: {skus: [...], khmer: str, active: bool, tier: int}}

    If brand_catalog.json exists, uses that. Otherwise builds from brands.py.
    """
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # Build from hardcoded data
    return _build_catalog_from_brands_py()


def _build_catalog_from_brands_py() -> dict:
    """Build catalog dict from the hardcoded brands.py constants."""
    catalog = {"brands": {}}
    for idx, (brand, skus) in enumerate(BRANDS_AND_SKUS.items(), start=1):
        catalog["brands"][brand] = {
            "skus": list(skus),
            "khmer": BRAND_KHMER.get(brand, ""),
            "active": True,
            "tier": 2,  # default to Tier 2
            "priority": idx,
        }
    return catalog


def save_brand_catalog(catalog: dict) -> None:
    """Save the catalog to brand_catalog.json."""
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def get_active_brands(catalog: dict | None = None) -> dict:
    """Return only active brands as {brand: [skus]} for use in the pipeline."""
    if catalog is None:
        catalog = load_brand_catalog()
    return {
        brand: info["skus"]
        for brand, info in catalog["brands"].items()
        if info.get("active", True)
    }


def get_active_brand_khmer(catalog: dict | None = None) -> dict:
    """Return Khmer names for active brands only."""
    if catalog is None:
        catalog = load_brand_catalog()
    return {
        brand: info["khmer"]
        for brand, info in catalog["brands"].items()
        if info.get("active", True) and info.get("khmer")
    }


def get_reference_image_counts() -> dict:
    """Count reference images per brand (flat directory, images named imageN.ext)."""
    ref_dir = Path(__file__).parent / "reference_images"
    if not ref_dir.exists():
        return {}
    # Reference images are flat-named, so we just return total count
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    total = sum(1 for f in ref_dir.iterdir() if f.suffix.lower() in extensions)
    return {"_total": total}


def get_annotation_counts() -> dict:
    """Count annotations per category from COCO annotations file."""
    coco_path = Path(__file__).parent / "training_data" / "annotations" / "coco_annotations.json"
    if not coco_path.exists():
        return {}
    try:
        with open(coco_path, "r") as f:
            coco = json.load(f)
        categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
        counts = {}
        for ann in coco.get("annotations", []):
            cat_name = categories.get(ann["category_id"], "unknown")
            counts[cat_name] = counts.get(cat_name, 0) + 1
        return counts
    except Exception:
        return {}
