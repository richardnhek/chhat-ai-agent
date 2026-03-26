"""
Crop Labeler — Fast brand labeling on auto-detected crops.

The model detects packs with RF-DETR, then the user labels each crop with the correct brand.
No drawing needed — just label and submit.
"""

import io
import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

# DINOv2 resolution patch + MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from brands import BRANDS_AND_SKUS
from corrections import save_correction, get_correction_stats
from auth import check_auth

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Labeler", page_icon="🏷️", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    .crop-grid-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem;
        border: 1px solid #e0e0e0; margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .crop-grid-card:hover { border-color: #4472C4; }
    .crop-grid-card.false-positive { opacity: 0.5; border-color: #dc3545; }

    .brand-badge {
        display: inline-block; background: #4472C4; color: white;
        padding: 0.2rem 0.6rem; border-radius: 16px;
        font-size: 0.8rem; font-weight: 500;
    }
    .confidence-badge {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .conf-high { background: #d4edda; color: #155724; }
    .conf-medium { background: #fff3cd; color: #856404; }
    .conf-low { background: #f8d7da; color: #721c24; }

    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }

    .progress-bar {
        background: #e8f4fd; border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;
    }
    .progress-text { font-weight: 600; color: #1a1a2e; }

    .result-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        margin-bottom: 1rem; border-left: 4px solid #4472C4;
    }
    .brand-tag {
        display: inline-block; background: #4472C4; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 500; margin: 0.15rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > div { flex: 1 1 100% !important; min-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── CHHAT Branding ────────────────────────────────────────────────────────
st.image("chhat-logo.png", width=120)
st.markdown("## Crop Labeler")
st.markdown("Fast brand labeling on auto-detected crops. No drawing needed.")

# ── Constants ─────────────────────────────────────────────────────────────
RFDETR_MODEL_PATH = "output/checkpoint_best_regular.pth"
BRAND_MODEL_PATH = "models/brand_classifier.pth"
SURVEY_DIR = Path("survey_images")
TRAINING_DIR = Path(__file__).parent.parent / "training_data"
ANNOTATIONS_DIR = TRAINING_DIR / "annotations"
IMAGES_DIR = TRAINING_DIR / "images"
COCO_FILE = ANNOTATIONS_DIR / "coco_annotations.json"

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BRAND_LIST = sorted(BRANDS_AND_SKUS.keys())
CATEGORY_MAP = {brand: idx + 1 for idx, brand in enumerate(BRAND_LIST)}


# ── Helpers ───────────────────────────────────────────────────────────────

def _check_rfdetr_model() -> bool:
    return Path(RFDETR_MODEL_PATH).exists()


def _apply_dinov2_patch():
    """Patch DINOv2 to handle non-standard resolutions."""
    try:
        import torch
        import torch.nn.functional as F

        def _patched_forward(self, x):
            B, C, H, W = x.shape
            patch_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size
            new_h = (H // patch_size) * patch_size
            new_w = (W // patch_size) * patch_size
            if new_h != H or new_w != W:
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            return self._original_forward(x)

        try:
            from timm.layers import PatchEmbed
            if not hasattr(PatchEmbed, '_original_forward'):
                PatchEmbed._original_forward = PatchEmbed.forward
                PatchEmbed.forward = _patched_forward
        except ImportError:
            pass
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading RF-DETR model...")
def _load_rfdetr():
    """Load the fine-tuned RF-DETR model."""
    _apply_dinov2_patch()
    try:
        from rfdetr import RFDETRBase
        model = RFDETRBase()
        model.load(RFDETR_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load RF-DETR: {e}")
        return None


def _run_detection(model, pil_image: Image.Image, threshold: float = 0.3) -> tuple[list[dict], float]:
    """Run RF-DETR detection on a PIL image."""
    start = time.time()
    try:
        detections = model.predict(pil_image, threshold=threshold)
        results = []
        for det in detections:
            box = det.xyxy
            if len(box) == 0:
                continue
            for i in range(len(box)):
                x1, y1, x2, y2 = int(box[i][0]), int(box[i][1]), int(box[i][2]), int(box[i][3])
                conf = float(det.confidence[i]) if hasattr(det, 'confidence') else 0.5

                crop_img = pil_image.crop((x1, y1, x2, y2))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)

                results.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "crop": crop_buf.getvalue(),
                    "crop_image": crop_img,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })
        elapsed = time.time() - start
        return results, elapsed
    except Exception as e:
        st.error(f"Detection error: {e}")
        return [], time.time() - start


def _classify_crops(detections: list[dict]) -> list[list[dict]]:
    """Run brand classification on all crops."""
    from brand_classifier import classify_crop
    classifications = []
    for det in detections:
        results = classify_crop(det["crop"], top_k=3)
        classifications.append(results)
    return classifications


def _load_coco() -> dict:
    """Load existing COCO annotations or return empty structure."""
    if COCO_FILE.exists():
        with open(COCO_FILE) as f:
            return json.load(f)
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()],
    }


def _save_coco(coco: dict):
    """Write COCO annotations to disk."""
    with open(COCO_FILE, "w") as f:
        json.dump(coco, f, indent=2)


def _next_id(items: list[dict]) -> int:
    """Return max id + 1, or 1 if list is empty."""
    if not items:
        return 1
    return max(item["id"] for item in items) + 1


# ── Sidebar Stats ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Labeling Stats")

    rfdetr_ok = _check_rfdetr_model()
    brand_ok = Path(BRAND_MODEL_PATH).exists()

    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{"Ready" if rfdetr_ok else "Missing"}</div>'
        f'  <div class="stat-lbl">RF-DETR Detector</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{"Ready" if brand_ok else "Missing"}</div>'
        f'  <div class="stat-lbl">Brand Classifier</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        stats = get_correction_stats()
        st.markdown(
            f'<div class="stat-box">'
            f'  <div class="stat-val">{stats.get("total", 0)}</div>'
            f'  <div class="stat-lbl">Total Corrections</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # COCO stats
    coco = _load_coco()
    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{len(coco["images"])}</div>'
        f'  <div class="stat-lbl">Labeled Images</div>'
        f'</div>'
        f'<div class="stat-box">'
        f'  <div class="stat-val">{len(coco["annotations"])}</div>'
        f'  <div class="stat-lbl">Total Bounding Boxes</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Keyboard Shortcuts")
    st.caption("**Tab** — Move to next crop")
    st.caption("**Enter** — Confirm current crop")


# ── Image Selection ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 1. Select Source")

survey_images = (sorted(SURVEY_DIR.glob("*.jpg")) + sorted(SURVEY_DIR.glob("*.jpeg")) +
                 sorted(SURVEY_DIR.glob("*.png"))) if SURVEY_DIR.exists() else []

image: Image.Image | None = None
image_source_name: str = ""

# Progress tracker for survey images
if survey_images:
    # Track which images have been labeled
    if "labeled_images" not in st.session_state:
        st.session_state["labeled_images"] = set()

    labeled_count = len(st.session_state["labeled_images"])
    total_count = len(survey_images)

    st.markdown(
        f'<div class="progress-bar">'
        f'  <span class="progress-text">Image {labeled_count} of {total_count} labeled</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Navigation
    if "current_img_idx" not in st.session_state:
        st.session_state["current_img_idx"] = 0

    col_prev, col_num, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous", use_container_width=True, key="prev_img",
                      disabled=st.session_state["current_img_idx"] <= 0):
            st.session_state["current_img_idx"] -= 1
            # Clear detection state for new image
            st.session_state.pop("cl_detections", None)
            st.session_state.pop("cl_classifications", None)
            st.rerun()
    with col_num:
        idx = st.number_input(
            "Image #", min_value=1, max_value=total_count,
            value=st.session_state["current_img_idx"] + 1,
            step=1, key="img_nav_num"
        )
        if idx - 1 != st.session_state["current_img_idx"]:
            st.session_state["current_img_idx"] = idx - 1
            st.session_state.pop("cl_detections", None)
            st.session_state.pop("cl_classifications", None)
            st.rerun()
    with col_next:
        if st.button("Next", use_container_width=True, key="next_img",
                      disabled=st.session_state["current_img_idx"] >= total_count - 1):
            st.session_state["current_img_idx"] += 1
            st.session_state.pop("cl_detections", None)
            st.session_state.pop("cl_classifications", None)
            st.rerun()

col_src1, col_src2 = st.columns(2)

with col_src1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="labeler_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name

with col_src2:
    if survey_images and not uploaded:
        img_idx = st.session_state["current_img_idx"]
        img_path = survey_images[img_idx]
        image = Image.open(img_path).convert("RGB")
        image_source_name = img_path.name
        st.success(f"**{image_source_name}** -- {image.size[0]}x{image.size[1]}px")
    elif not survey_images and not uploaded:
        st.info("No images in survey_images/ folder.")

if image is None:
    st.info("Upload an image or select one from survey_images/ to begin labeling.")
    st.stop()


# ── Auto-detect Packs ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2. Auto-Detect Packs")

use_rfdetr = rfdetr_ok
fallback_mode = False

if not use_rfdetr:
    st.warning("RF-DETR model not available. Falling back to manual mode -- select brands for the full image.")
    fallback_mode = True

# Detection / Fallback
if not fallback_mode:
    conf_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.3, 0.05, key="cl_threshold")

    detect_key = f"detect_{image_source_name}"
    if st.button("Detect Packs", type="primary", use_container_width=True, key="cl_detect"):
        with st.spinner("Running RF-DETR detection..."):
            model = _load_rfdetr()
            if model is None:
                st.error("Failed to load RF-DETR model.")
                st.stop()

            detections, det_time = _run_detection(model, image, threshold=conf_threshold)
            st.session_state["cl_detections"] = detections
            st.session_state["cl_det_time"] = det_time
            st.session_state["cl_det_image"] = image_source_name

            # Auto-classify if brand model exists
            if detections and brand_ok:
                with st.spinner("Classifying brands..."):
                    classifications = _classify_crops(detections)
                    st.session_state["cl_classifications"] = classifications
            else:
                st.session_state["cl_classifications"] = [[] for _ in detections]

    if "cl_detections" not in st.session_state or st.session_state.get("cl_det_image") != image_source_name:
        st.info("Click **Detect Packs** to auto-detect cigarette packs in this image.")
        st.stop()

    detections = st.session_state["cl_detections"]
    classifications = st.session_state.get("cl_classifications", [[] for _ in detections])
    det_time = st.session_state.get("cl_det_time", 0)

    if not detections:
        st.warning("No packs detected. Try lowering the confidence threshold.")
        st.stop()

    st.success(f"Detected **{len(detections)}** packs in {det_time:.2f}s")

    # ── Crop Grid ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. Label Crops")
    st.caption("Review each detected crop. Correct the brand if needed, select SKU, and mark false positives.")

    # Initialize form state
    if "crop_labels" not in st.session_state or st.session_state.get("cl_labels_image") != image_source_name:
        crop_labels = []
        for i, det in enumerate(detections):
            cls_list = classifications[i] if i < len(classifications) else []
            predicted_brand = cls_list[0]["brand"] if cls_list else ""
            predicted_conf = cls_list[0]["confidence"] if cls_list else 0
            crop_labels.append({
                "brand": predicted_brand,
                "sku": "",
                "not_cigarette": False,
                "predicted_brand": predicted_brand,
                "predicted_conf": predicted_conf,
            })
        st.session_state["crop_labels"] = crop_labels
        st.session_state["cl_labels_image"] = image_source_name

    crop_labels = st.session_state["crop_labels"]

    # Display crops in grid
    cols_per_row = 3
    for row_start in range(0, len(detections), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, det_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(detections)))):
            det = detections[det_idx]
            label_data = crop_labels[det_idx]
            cls_list = classifications[det_idx] if det_idx < len(classifications) else []

            with cols[col_idx]:
                # Crop image
                st.image(det["crop"], caption=f"Crop #{det_idx+1} ({det['width']}x{det['height']}px)", width=220)

                # Model prediction badge
                if cls_list:
                    top = cls_list[0]
                    conf_level = "conf-high" if top["confidence"] >= 0.7 else ("conf-medium" if top["confidence"] >= 0.4 else "conf-low")
                    st.markdown(
                        f'<span class="brand-badge">{top["brand"]}</span> '
                        f'<span class="confidence-badge {conf_level}">{top["confidence"]:.0%}</span>',
                        unsafe_allow_html=True,
                    )
                    # Show top-3
                    with st.expander("Top-3"):
                        for rank, pred in enumerate(cls_list, 1):
                            st.write(f"{rank}. {pred['brand']} -- {pred['confidence']:.1%}")

                # Brand dropdown
                brand_idx = BRAND_LIST.index(label_data["brand"]) + 1 if label_data["brand"] in BRAND_LIST else 0
                selected_brand = st.selectbox(
                    f"Brand #{det_idx+1}",
                    ["-- Select --"] + BRAND_LIST,
                    index=brand_idx,
                    key=f"brand_{det_idx}",
                )
                if selected_brand != "-- Select --":
                    crop_labels[det_idx]["brand"] = selected_brand
                else:
                    crop_labels[det_idx]["brand"] = ""

                # SKU dropdown (filtered by brand)
                current_brand = crop_labels[det_idx]["brand"]
                if current_brand and current_brand in BRANDS_AND_SKUS:
                    skus = BRANDS_AND_SKUS[current_brand]
                    sku_idx = skus.index(label_data["sku"]) + 1 if label_data["sku"] in skus else 0
                    selected_sku = st.selectbox(
                        f"SKU #{det_idx+1}",
                        ["-- Select --"] + skus,
                        index=sku_idx,
                        key=f"sku_{det_idx}",
                    )
                    crop_labels[det_idx]["sku"] = selected_sku if selected_sku != "-- Select --" else ""

                # Not a cigarette checkbox
                not_cig = st.checkbox(
                    "Not a cigarette pack",
                    value=label_data["not_cigarette"],
                    key=f"notcig_{det_idx}",
                )
                crop_labels[det_idx]["not_cigarette"] = not_cig

    st.session_state["crop_labels"] = crop_labels


    # ── Submit All ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Submit Labels")

    # Summary before submit
    valid_labels = [cl for cl in crop_labels if cl["brand"] and not cl["not_cigarette"]]
    false_positives = [cl for cl in crop_labels if cl["not_cigarette"]]
    unlabeled = [cl for cl in crop_labels if not cl["brand"] and not cl["not_cigarette"]]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Labeled Crops", len(valid_labels))
    with col_s2:
        st.metric("False Positives", len(false_positives))
    with col_s3:
        st.metric("Unlabeled", len(unlabeled))

    if unlabeled:
        st.warning(f"{len(unlabeled)} crop(s) still unlabeled. Label them or mark as false positives before submitting.")

    if st.button("Submit All Labels", type="primary", use_container_width=True, key="submit_labels",
                  disabled=len(unlabeled) > 0):
        img_w, img_h = image.size
        serial = image_source_name.replace("serial_", "").split("_")[0]

        # Save corrections for crops where model was wrong
        corrections_saved = 0
        for i, det in enumerate(detections):
            label = crop_labels[i]
            if label["not_cigarette"]:
                # Save as false positive correction
                save_correction({
                    "serial": serial,
                    "image_url": image_source_name,
                    "model_used": "rfdetr+brand_classifier",
                    "ai_result": {"brands": [label["predicted_brand"]] if label["predicted_brand"] else [], "skus": []},
                    "corrected_result": {"brands": [], "skus": []},
                    "notes": json.dumps({
                        "crop_index": i,
                        "bbox": det["box"],
                        "false_positive": True,
                    }),
                })
                corrections_saved += 1
            elif label["brand"] and label["brand"] != label["predicted_brand"]:
                save_correction({
                    "serial": serial,
                    "image_url": image_source_name,
                    "model_used": "rfdetr+brand_classifier",
                    "ai_result": {"brands": [label["predicted_brand"]] if label["predicted_brand"] else [], "skus": []},
                    "corrected_result": {"brands": [label["brand"]], "skus": [label["sku"]] if label["sku"] else []},
                    "notes": json.dumps({
                        "crop_index": i,
                        "bbox": det["box"],
                        "detection_confidence": det["confidence"],
                    }),
                })
                corrections_saved += 1

        # Save COCO annotations for valid (non-false-positive) crops
        coco = _load_coco()
        image_id = _next_id(coco["images"])
        coco["images"].append({
            "id": image_id,
            "file_name": image_source_name,
            "width": img_w,
            "height": img_h,
        })

        boxes_saved = 0
        for i, det in enumerate(detections):
            label = crop_labels[i]
            if label["not_cigarette"] or not label["brand"]:
                continue

            x1, y1, x2, y2 = det["box"]
            w = x2 - x1
            h = y2 - y1
            cat_id = CATEGORY_MAP.get(label["brand"], 1)

            ann_id = _next_id(coco["annotations"])
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "brand": label["brand"],
                "sku": label.get("sku", ""),
            })
            boxes_saved += 1

        _save_coco(coco)

        # Save image to training data
        image.save(str(IMAGES_DIR / image_source_name), "JPEG", quality=95)

        # Mark image as labeled
        if "labeled_images" in st.session_state:
            st.session_state["labeled_images"].add(image_source_name)

        # Summary
        brands_identified = set(cl["brand"] for cl in valid_labels)
        st.success(
            f"**{len(valid_labels)} crops labeled**, {len(brands_identified)} brands identified. "
            f"{corrections_saved} corrections saved. {boxes_saved} COCO annotations added."
        )
        st.balloons()

        # Clear state for this image
        st.session_state.pop("cl_detections", None)
        st.session_state.pop("cl_classifications", None)
        st.session_state.pop("crop_labels", None)

else:
    # ── Fallback Mode (no RF-DETR) ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Manual Brand Selection")
    st.caption("Since RF-DETR is not available, select the brands visible in the full image.")

    st.image(image, caption=image_source_name, use_container_width=True)

    selected_brands = st.multiselect("Brands visible in this image", BRAND_LIST, key="fallback_brands")

    sku_selections = {}
    if selected_brands:
        st.markdown("#### Select SKUs for each brand")
        for brand in selected_brands:
            skus = BRANDS_AND_SKUS.get(brand, [])
            selected_skus = st.multiselect(f"SKUs for {brand}", skus, key=f"fallback_sku_{brand}")
            sku_selections[brand] = selected_skus

    if selected_brands and st.button("Save Labels", type="primary", use_container_width=True, key="fallback_save"):
        serial = image_source_name.replace("serial_", "").split("_")[0]
        all_skus = [sku for skus in sku_selections.values() for sku in skus]

        save_correction({
            "serial": serial,
            "image_url": image_source_name,
            "model_used": "human_annotation_fallback",
            "ai_result": {"brands": [], "skus": []},
            "corrected_result": {"brands": selected_brands, "skus": all_skus},
            "notes": "Manual labeling (no RF-DETR)",
        })

        # Save to COCO with full-image bbox
        img_w, img_h = image.size
        coco = _load_coco()
        image_id = _next_id(coco["images"])
        coco["images"].append({
            "id": image_id,
            "file_name": image_source_name,
            "width": img_w,
            "height": img_h,
        })

        for brand in selected_brands:
            ann_id = _next_id(coco["annotations"])
            cat_id = CATEGORY_MAP.get(brand, 1)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [0, 0, img_w, img_h],
                "area": img_w * img_h,
                "iscrowd": 0,
                "brand": brand,
            })

        _save_coco(coco)
        image.save(str(IMAGES_DIR / image_source_name), "JPEG", quality=95)

        if "labeled_images" in st.session_state:
            st.session_state["labeled_images"].add(image_source_name)

        st.success(f"Saved {len(selected_brands)} brands for this image!")
        st.balloons()
