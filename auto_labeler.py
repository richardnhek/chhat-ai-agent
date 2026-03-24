"""
Auto-Labeler — Uses Gemini to generate bounding box annotations for training.

This is used ONLY during training data preparation, NOT in production.
Gemini analyzes images and outputs bounding box coordinates + brand labels
in COCO format, ready for RF-DETR training.

Usage:
    python auto_labeler.py image1.jpg image2.jpg --output training_data/auto_labeled/
    python auto_labeler.py --dir survey_photos/ --output training_data/auto_labeled/
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from brands import BRANDS_AND_SKUS
from image_analyzer import _resize_image
from rate_limiter import RateLimiter

load_dotenv()

LABELING_PROMPT = """Analyze this image of a store display case. For every cigarette pack/box visible, provide:
1. A bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
2. The brand name (from the official list below)
3. The SKU if identifiable

Official brands: """ + ", ".join(sorted(BRANDS_AND_SKUS.keys())) + """

Respond in this EXACT JSON format (no markdown):
{
    "image_width": <width in pixels>,
    "image_height": <height in pixels>,
    "detections": [
        {"box": [x_min, y_min, x_max, y_max], "brand": "<brand>", "sku": "<sku or null>", "confidence": "<high|medium|low>"},
    ]
}

If no cigarette packs are visible, return: {"image_width": 0, "image_height": 0, "detections": []}

Draw bounding boxes tightly around each individual pack. Include partially visible packs.
"""


def auto_label_image(
    image_data: bytes,
    model: str = "gemini-2.5-pro",
    api_keys: dict | None = None,
) -> dict:
    """
    Use Gemini to generate bounding box annotations for a single image.
    Returns detection results with bounding boxes.
    """
    from image_analyzer import analyze_image

    resized_data, media_type = _resize_image(image_data, "image/jpeg")

    # Get image dimensions
    from PIL import Image
    img = Image.open(io.BytesIO(resized_data))
    img_w, img_h = img.size

    result = analyze_image(
        resized_data, media_type,
        model=model,
        api_keys=api_keys,
        correction_context=LABELING_PROMPT,
    )

    # Parse bounding boxes from result
    detections = []
    if "brands_found" in result:
        # Standard format — no bounding boxes, convert to approximate boxes
        for entry in result.get("brands_found", []):
            brand = entry.get("brand", "")
            if brand in BRANDS_AND_SKUS:
                detections.append({
                    "brand": brand,
                    "sku": entry.get("skus", [None])[0] if entry.get("skus") else None,
                    "confidence": "medium",
                    "box": None,  # No box available from standard prompt
                })
    elif "detections" in result:
        # Auto-label format with bounding boxes
        for det in result["detections"]:
            brand = det.get("brand", "")
            if brand in BRANDS_AND_SKUS:
                detections.append(det)

    return {
        "image_width": img_w,
        "image_height": img_h,
        "detections": detections,
    }


def auto_label_batch(
    image_paths: list[str],
    output_dir: str = "training_data/auto_labeled",
    model: str = "gemini-2.5-pro",
) -> dict:
    """
    Auto-label a batch of images and output COCO format annotations.
    """
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "claude": os.getenv("ANTHROPIC_API_KEY", ""),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    limiter = RateLimiter(model)

    # Build COCO structure
    categories = []
    brand_to_id = {}
    for i, brand in enumerate(sorted(BRANDS_AND_SKUS.keys()), 1):
        categories.append({"id": i, "name": brand})
        brand_to_id[brand] = i

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    labeled_count = 0
    total_boxes = 0

    for img_idx, img_path in enumerate(image_paths):
        try:
            with open(img_path, "rb") as f:
                image_data = f.read()

            # Copy image to output
            filename = f"img_{img_idx:05d}{Path(img_path).suffix}"
            (images_dir / filename).write_bytes(image_data)

            # Auto-label
            limiter.wait()
            result = auto_label_image(image_data, model=model, api_keys=api_keys)

            img_w = result["image_width"]
            img_h = result["image_height"]

            coco["images"].append({
                "id": img_idx + 1,
                "file_name": filename,
                "width": img_w,
                "height": img_h,
            })

            for det in result["detections"]:
                brand = det.get("brand", "")
                if brand not in brand_to_id:
                    continue

                box = det.get("box")
                if box and len(box) == 4:
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    if w > 0 and h > 0:
                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_idx + 1,
                            "category_id": brand_to_id[brand],
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        })
                        annotation_id += 1
                        total_boxes += 1

            labeled_count += 1
            print(f"  [{img_idx+1}/{len(image_paths)}] {Path(img_path).name}: "
                  f"{len(result['detections'])} detections")

        except Exception as e:
            print(f"  [{img_idx+1}/{len(image_paths)}] {Path(img_path).name}: ERROR — {e}")

    # Save COCO annotations
    annotations_path = output_path / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone! {labeled_count} images labeled, {total_boxes} bounding boxes")
    print(f"Output: {annotations_path}")

    return {
        "images_labeled": labeled_count,
        "total_boxes": total_boxes,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using Gemini for RF-DETR training")
    parser.add_argument("images", nargs="*", help="Image files to label")
    parser.add_argument("--dir", help="Directory of images to label")
    parser.add_argument("--output", default="training_data/auto_labeled", help="Output directory")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model to use for labeling")
    args = parser.parse_args()

    image_paths = list(args.images)
    if args.dir:
        dir_path = Path(args.dir)
        image_paths.extend(str(p) for p in dir_path.glob("*.jpg"))
        image_paths.extend(str(p) for p in dir_path.glob("*.jpeg"))
        image_paths.extend(str(p) for p in dir_path.glob("*.png"))

    if not image_paths:
        print("No images specified. Use: python auto_labeler.py image.jpg or --dir folder/")
        sys.exit(1)

    print(f"Auto-labeling {len(image_paths)} images with {args.model}...")
    auto_label_batch(image_paths, output_dir=args.output, model=args.model)
