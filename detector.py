"""
Pack Detector — RF-DETR based cigarette pack detection.

Stage 1 of the hybrid pipeline:
  Image → Detect all cigarette packs → Return bounding boxes + crops

Two modes:
  - Pre-trained (class-agnostic): finds rectangular objects that look like cigarette packs
  - Fine-tuned (brand-aware): finds packs AND classifies by brand
"""

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

# RF-DETR will be imported lazily to handle installation gracefully
_rfdetr_model = None
_model_lock = None


def _get_model(model_path: str | None = None):
    """Lazy-load the RF-DETR model."""
    global _rfdetr_model, _model_lock

    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    with _model_lock:
        if _rfdetr_model is None:
            try:
                from rfdetr import RFDETRBase, RFDETRLarge

                if model_path and Path(model_path).exists():
                    # Load fine-tuned model
                    _rfdetr_model = RFDETRBase()
                    _rfdetr_model.load(model_path)
                else:
                    # Use pre-trained COCO model
                    _rfdetr_model = RFDETRBase()

            except ImportError:
                return None

    return _rfdetr_model


def detect_packs(
    image_data: bytes,
    confidence_threshold: float = 0.3,
    model_path: str | None = None,
) -> list[dict]:
    """
    Detect cigarette packs in an image.

    Returns list of detections:
    [
        {
            "box": [x1, y1, x2, y2],  # pixel coordinates
            "confidence": 0.87,
            "class": "cigarette_pack",  # or brand name if fine-tuned
            "crop": bytes,  # cropped image of just this pack
        }
    ]
    """
    model = _get_model(model_path)

    if model is None:
        # RF-DETR not available — fall back to full-image mode
        return _fallback_detect(image_data)

    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_data))
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Run detection
    try:
        detections = model.predict(img, threshold=confidence_threshold)

        results = []
        img_array = np.array(img)

        for det in detections:
            box = det.xyxy
            if len(box) == 0:
                continue

            for i in range(len(box)):
                x1, y1, x2, y2 = int(box[i][0]), int(box[i][1]), int(box[i][2]), int(box[i][3])
                conf = float(det.confidence[i]) if hasattr(det, 'confidence') else 0.5
                cls = str(det.data.get('class_name', ['object'])[i]) if hasattr(det, 'data') else 'object'

                # Crop the detected region
                crop_img = img.crop((x1, y1, x2, y2))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)
                crop_bytes = crop_buf.getvalue()

                results.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls,
                    "crop": crop_bytes,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })

        return results

    except Exception as e:
        return _fallback_detect(image_data)


def _fallback_detect(image_data: bytes) -> list[dict]:
    """
    Fallback when RF-DETR is not available:
    Return the full image as a single "detection" so downstream
    processing still works (just less accurate).
    """
    return [{
        "box": [0, 0, 0, 0],
        "confidence": 1.0,
        "class": "full_image",
        "crop": image_data,
        "width": 0,
        "height": 0,
        "is_fallback": True,
    }]


def crop_image_region(image_data: bytes, box: list[int], padding: int = 10) -> bytes:
    """Crop a region from an image with optional padding."""
    img = Image.open(io.BytesIO(image_data))
    x1, y1, x2, y2 = box

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)

    crop = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
