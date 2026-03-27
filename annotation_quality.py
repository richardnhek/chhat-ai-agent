"""
Annotation Quality Checker — validates COCO-format training data and flags issues.

Provides check_annotation_quality() for diagnostics and auto_fix_annotations() for
automated cleanup of common annotation problems.
"""

import json
import copy
from collections import Counter, defaultdict
from pathlib import Path


def check_annotation_quality(
    coco_path: str = "training_data/annotations/coco_annotations.json",
) -> dict:
    """
    Validate annotations and flag issues.

    Checks:
    1. Boxes too small (< 10x10 pixels) — likely misclicks
    2. Boxes too large (> 80% of image area) — likely wrong
    3. Boxes with zero or negative dimensions
    4. Duplicate boxes (same position +/- 5px on same image)
    5. Images with too many boxes (> 50) — suspicious
    6. Images with 0 boxes — should they be included?
    7. Boxes outside image bounds
    8. Class distribution imbalance

    Returns dict with:
    - total_issues: int
    - issues: list of dicts
    - summary: counts per issue type
    - class_distribution: category counts
    """
    path = Path(coco_path)
    if not path.exists():
        return {
            "total_issues": 0,
            "issues": [],
            "summary": {},
            "class_distribution": {},
            "error": f"File not found: {coco_path}",
        }

    with open(path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    issues = []
    summary = Counter()
    class_counts = Counter()

    # Index annotations per image
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)
        class_counts[categories.get(ann["category_id"], f"id_{ann['category_id']}")] += 1

    # --- Check each annotation ---
    for ann in annotations:
        img = images.get(ann["image_id"])
        if img is None:
            issues.append({
                "image": f"image_id={ann['image_id']}",
                "annotation_id": ann["id"],
                "issue_type": "missing_image",
                "details": "Annotation references non-existent image",
            })
            summary["missing_image"] += 1
            continue

        x, y, w, h = ann["bbox"]
        img_w, img_h = img["width"], img["height"]
        fname = img.get("file_name", f"id={img['id']}")

        # Check 3: zero or negative dimensions
        if w <= 0 or h <= 0:
            issues.append({
                "image": fname,
                "annotation_id": ann["id"],
                "issue_type": "invalid_dimensions",
                "details": f"Box has zero/negative size: {w}x{h}",
            })
            summary["invalid_dimensions"] += 1
            continue

        # Check 1: too small
        if w < 10 and h < 10:
            issues.append({
                "image": fname,
                "annotation_id": ann["id"],
                "issue_type": "too_small",
                "details": f"Box {w}x{h} is smaller than 10x10",
            })
            summary["too_small"] += 1

        # Check 2: too large
        box_area = w * h
        img_area = img_w * img_h
        if img_area > 0 and box_area > 0.8 * img_area:
            issues.append({
                "image": fname,
                "annotation_id": ann["id"],
                "issue_type": "too_large",
                "details": f"Box covers {box_area / img_area * 100:.1f}% of image ({w}x{h} on {img_w}x{img_h})",
            })
            summary["too_large"] += 1

        # Check 7: out of bounds
        if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
            issues.append({
                "image": fname,
                "annotation_id": ann["id"],
                "issue_type": "out_of_bounds",
                "details": f"Box [{x},{y},{w},{h}] exceeds image bounds {img_w}x{img_h}",
            })
            summary["out_of_bounds"] += 1

    # Check 4: duplicates (same image, same category, position within 5px)
    for img_id, img_anns in anns_by_image.items():
        fname = images[img_id].get("file_name", f"id={img_id}") if img_id in images else f"id={img_id}"
        seen = []
        for ann in img_anns:
            bx, by, bw, bh = ann["bbox"]
            for prev_ann, (px, py, pw, ph) in seen:
                if (
                    ann["category_id"] == prev_ann["category_id"]
                    and abs(bx - px) <= 5
                    and abs(by - py) <= 5
                    and abs(bw - pw) <= 5
                    and abs(bh - ph) <= 5
                ):
                    issues.append({
                        "image": fname,
                        "annotation_id": ann["id"],
                        "issue_type": "duplicate",
                        "details": f"Near-duplicate of annotation {prev_ann['id']}",
                    })
                    summary["duplicate"] += 1
                    break
            seen.append((ann, (bx, by, bw, bh)))

        # Check 5: too many boxes
        if len(img_anns) > 50:
            issues.append({
                "image": fname,
                "annotation_id": None,
                "issue_type": "too_many_boxes",
                "details": f"Image has {len(img_anns)} annotations (> 50)",
            })
            summary["too_many_boxes"] += 1

    # Check 6: images with 0 annotations
    for img_id, img in images.items():
        if img_id not in anns_by_image:
            issues.append({
                "image": img.get("file_name", f"id={img_id}"),
                "annotation_id": None,
                "issue_type": "no_annotations",
                "details": "Image has no annotations",
            })
            summary["no_annotations"] += 1

    # Check 8: class imbalance — flag categories with < 10% of median count
    if class_counts:
        counts = list(class_counts.values())
        median_count = sorted(counts)[len(counts) // 2]
        threshold = max(1, median_count * 0.1)
        for cat_name, count in class_counts.items():
            if count < threshold:
                issues.append({
                    "image": None,
                    "annotation_id": None,
                    "issue_type": "class_imbalance",
                    "details": f"'{cat_name}' has only {count} annotations (median is {median_count})",
                })
                summary["class_imbalance"] += 1

    return {
        "total_issues": len(issues),
        "issues": issues,
        "summary": dict(summary),
        "class_distribution": dict(class_counts),
        "total_images": len(images),
        "total_annotations": len(annotations),
        "total_categories": len(categories),
    }


def auto_fix_annotations(
    coco_path: str = "training_data/annotations/coco_annotations.json",
    fixes: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Auto-fix common annotation issues.

    Supported fixes:
    - remove_tiny: delete boxes < 10x10
    - clip_bounds: clip boxes to image dimensions
    - remove_duplicates: remove near-duplicate boxes

    Returns count of fixes applied per type, and saves the fixed file.
    """
    if fixes is None:
        fixes = ["remove_tiny", "clip_bounds"]

    path = Path(coco_path)
    if not path.exists():
        return {"error": f"File not found: {coco_path}", "fixes_applied": {}}

    with open(path, "r") as f:
        coco = json.load(f)

    coco_fixed = copy.deepcopy(coco)
    images = {img["id"]: img for img in coco_fixed.get("images", [])}
    annotations = coco_fixed.get("annotations", [])
    fix_counts = Counter()
    ids_to_remove = set()

    # --- remove_tiny ---
    if "remove_tiny" in fixes:
        for ann in annotations:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if w < 10 and h < 10:
                ids_to_remove.add(ann["id"])
                fix_counts["remove_tiny"] += 1

    # --- clip_bounds ---
    if "clip_bounds" in fixes:
        for ann in annotations:
            if ann["id"] in ids_to_remove:
                continue
            img = images.get(ann["image_id"])
            if img is None:
                continue
            x, y, w, h = ann["bbox"]
            img_w, img_h = img["width"], img["height"]
            new_x = max(0, x)
            new_y = max(0, y)
            new_w = min(w, img_w - new_x)
            new_h = min(h, img_h - new_y)
            if (new_x, new_y, new_w, new_h) != (x, y, w, h):
                ann["bbox"] = [new_x, new_y, new_w, new_h]
                ann["area"] = new_w * new_h
                fix_counts["clip_bounds"] += 1

    # --- remove_duplicates ---
    if "remove_duplicates" in fixes:
        anns_by_image = defaultdict(list)
        for ann in annotations:
            if ann["id"] not in ids_to_remove:
                anns_by_image[ann["image_id"]].append(ann)

        for img_id, img_anns in anns_by_image.items():
            seen = []
            for ann in img_anns:
                bx, by, bw, bh = ann["bbox"]
                is_dup = False
                for prev_ann, (px, py, pw, ph) in seen:
                    if (
                        ann["category_id"] == prev_ann["category_id"]
                        and abs(bx - px) <= 5
                        and abs(by - py) <= 5
                        and abs(bw - pw) <= 5
                        and abs(bh - ph) <= 5
                    ):
                        ids_to_remove.add(ann["id"])
                        fix_counts["remove_duplicates"] += 1
                        is_dup = True
                        break
                if not is_dup:
                    seen.append((ann, (bx, by, bw, bh)))

    # Apply removals
    coco_fixed["annotations"] = [
        ann for ann in coco_fixed["annotations"] if ann["id"] not in ids_to_remove
    ]

    # Save
    if output_path is None:
        output_path = coco_path
    with open(output_path, "w") as f:
        json.dump(coco_fixed, f, indent=2)

    return {
        "fixes_applied": dict(fix_counts),
        "total_fixes": sum(fix_counts.values()),
        "annotations_removed": len(ids_to_remove),
        "annotations_remaining": len(coco_fixed["annotations"]),
        "output_path": str(output_path),
    }
