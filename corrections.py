"""
Correction loop — stores human corrections and retrieves relevant ones as few-shot examples.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

CORRECTIONS_FILE = "corrections.json"


def load_corrections(path: str = CORRECTIONS_FILE) -> list[dict]:
    """Load all corrections from the JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_correction(correction: dict, path: str = CORRECTIONS_FILE):
    """Append a correction record to the JSON file."""
    corrections = load_corrections(path)

    correction["id"] = str(uuid.uuid4())[:8]
    correction["timestamp"] = datetime.now().isoformat()

    # Auto-classify correction type
    ai_brands = set(correction.get("ai_result", {}).get("brands", []))
    corrected_brands = set(correction.get("corrected_result", {}).get("brands", []))

    if ai_brands == corrected_brands:
        ai_skus = set(correction.get("ai_result", {}).get("skus", []))
        corrected_skus = set(correction.get("corrected_result", {}).get("skus", []))
        if ai_skus == corrected_skus:
            correction["correction_type"] = "no_change"
        else:
            correction["correction_type"] = "sku_corrected"
    else:
        added = corrected_brands - ai_brands
        removed = ai_brands - corrected_brands
        if added and removed:
            correction["correction_type"] = "brand_swap"
        elif added:
            correction["correction_type"] = "brand_added"
        elif removed:
            correction["correction_type"] = "brand_removed"

    corrections.append(correction)

    with open(path, "w") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)


def save_corrections_batch(corrections_list: list[dict], path: str = CORRECTIONS_FILE):
    """Save multiple corrections at once."""
    for c in corrections_list:
        if c.get("correction_type") != "no_change":
            save_correction(c, path)


def find_relevant_corrections(
    ai_brands: list[str] | None = None,
    limit: int = 5,
    path: str = CORRECTIONS_FILE,
) -> list[dict]:
    """
    Find the most relevant past corrections to inject as few-shot examples.

    Relevance is determined by brand overlap with the AI's current detection.
    If no AI brands provided (first pass), returns the most recent corrections.
    Only returns corrections where actual changes were made.
    """
    corrections = load_corrections(path)

    # Filter out no_change records
    corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    if not corrections:
        return []

    if not ai_brands:
        # Cold start: return most recent corrections
        corrections.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
        return corrections[:limit]

    # Score by brand overlap
    ai_set = set(ai_brands)
    scored = []
    for c in corrections:
        ai_result_brands = set(c.get("ai_result", {}).get("brands", []))
        corrected_brands = set(c.get("corrected_result", {}).get("brands", []))
        all_brands_in_correction = ai_result_brands | corrected_brands

        overlap = len(ai_set & all_brands_in_correction)
        scored.append((overlap, c))

    # Sort by overlap (desc), then by recency (desc)
    scored.sort(key=lambda x: (x[0], x[1].get("timestamp", "")), reverse=True)

    return [c for _, c in scored[:limit]]


def format_corrections_for_prompt(corrections: list[dict]) -> str:
    """Format corrections as few-shot examples for the prompt."""
    if not corrections:
        return ""

    lines = [
        "LEARNING FROM PAST CORRECTIONS:",
        "The following are real examples where the AI made mistakes that were corrected by human reviewers.",
        "Use these to avoid repeating the same errors:\n",
    ]

    for i, c in enumerate(corrections, 1):
        ai = c.get("ai_result", {})
        corrected = c.get("corrected_result", {})
        notes = c.get("notes", "")
        correction_type = c.get("correction_type", "unknown")

        ai_brands = ", ".join(ai.get("brands", [])) or "None"
        ai_skus = ", ".join(ai.get("skus", [])) or "None"
        correct_brands = ", ".join(corrected.get("brands", [])) or "None"
        correct_skus = ", ".join(corrected.get("skus", [])) or "None"

        lines.append(f"Correction {i} ({correction_type}):")
        lines.append(f"  AI detected brands: {ai_brands}")
        lines.append(f"  AI detected SKUs: {ai_skus}")
        lines.append(f"  CORRECT brands: {correct_brands}")
        lines.append(f"  CORRECT SKUs: {correct_skus}")
        if notes:
            lines.append(f"  Reviewer note: \"{notes}\"")
        lines.append("")

    return "\n".join(lines)


def get_correction_stats(path: str = CORRECTIONS_FILE) -> dict:
    """Get summary stats about corrections."""
    corrections = load_corrections(path)
    real_corrections = [c for c in corrections if c.get("correction_type") != "no_change"]

    return {
        "total": len(real_corrections),
        "brand_swaps": len([c for c in real_corrections if c.get("correction_type") == "brand_swap"]),
        "brand_added": len([c for c in real_corrections if c.get("correction_type") == "brand_added"]),
        "brand_removed": len([c for c in real_corrections if c.get("correction_type") == "brand_removed"]),
        "sku_corrected": len([c for c in real_corrections if c.get("correction_type") == "sku_corrected"]),
    }
