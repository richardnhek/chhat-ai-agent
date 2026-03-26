"""
Model Tester — Test any image against the current RF-DETR + brand classifier pipeline.

Upload or select an image, run detection, classify brands, compare with ground truth,
and quickly correct any mistakes.
"""

import io
import os
import json
import time
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
st.set_page_config(page_title="Model Tester", page_icon="🧪", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    .result-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        margin-bottom: 1rem; border-left: 4px solid #4472C4;
    }
    .result-card.success { border-left-color: #28a745; }
    .result-card.error { border-left-color: #dc3545; }
    .result-card.warn { border-left-color: #FF8C00; }

    .brand-tag {
        display: inline-block; background: #4472C4; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 500; margin: 0.15rem;
    }
    .confidence-high { color: #28a745; font-weight: 700; }
    .confidence-medium { color: #FF8C00; font-weight: 700; }
    .confidence-low { color: #dc3545; font-weight: 700; }

    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }

    .crop-card {
        background: #f8f9fa; border-radius: 10px; padding: 0.8rem;
        border: 1px solid #e0e0e0; margin-bottom: 0.8rem;
    }
    .crop-card.mismatch { border: 2px solid #dc3545; background: #fff5f5; }

    .diff-added { background: #d4edda; border-radius: 4px; padding: 0.2rem 0.5rem; }
    .diff-removed { background: #f8d7da; border-radius: 4px; padding: 0.2rem 0.5rem; text-decoration: line-through; }

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
st.markdown("## Model Tester")
st.markdown("Test any image against the current RF-DETR detector and brand classifier.")

# ── Constants ─────────────────────────────────────────────────────────────
RFDETR_MODEL_PATH = "output/checkpoint_best_regular.pth"
BRAND_MODEL_PATH = "models/brand_classifier.pth"
SURVEY_DIR = Path("survey_images")
TRAINING_DIR = Path(__file__).parent.parent / "training_data"
ANNOTATIONS_DIR = TRAINING_DIR / "annotations"
COCO_FILE = ANNOTATIONS_DIR / "coco_annotations.json"

BRAND_LIST = sorted(BRANDS_AND_SKUS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────

def _check_rfdetr_model() -> bool:
    return Path(RFDETR_MODEL_PATH).exists()


def _check_brand_model() -> bool:
    return Path(BRAND_MODEL_PATH).exists()


def _apply_dinov2_patch():
    """Patch DINOv2 to handle non-standard resolutions."""
    try:
        import torch
        import torch.nn.functional as F

        def _patched_forward(self, x):
            B, C, H, W = x.shape
            patch_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size
            # Resize to be divisible by patch_size
            new_h = (H // patch_size) * patch_size
            new_w = (W // patch_size) * patch_size
            if new_h != H or new_w != W:
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            return self._original_forward(x)

        # Attempt to patch if the model uses DINOv2 backbone
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


def _run_detection(model, pil_image: Image.Image, threshold: float = 0.3) -> list[dict]:
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


def _draw_boxes(pil_image: Image.Image, detections: list[dict], labels: list[str] = None) -> Image.Image:
    """Draw bounding boxes on image."""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    colors = ["#FF4444", "#44AA44", "#4444FF", "#FF8800", "#8844FF", "#FF44AA",
              "#44DDDD", "#AAAA00", "#00AAAA", "#AA44AA"]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = labels[i] if labels and i < len(labels) else f"#{i+1}"
        conf = det.get("confidence", 0)
        text = f"{label} ({conf:.0%})"

        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1), text, fill="white", font=font)

    return img


def _load_ground_truth(image_name: str) -> list[dict] | None:
    """Check if image has annotations in corrections or COCO data."""
    # Check COCO annotations
    if COCO_FILE.exists():
        try:
            with open(COCO_FILE) as f:
                coco = json.load(f)
            for img_entry in coco.get("images", []):
                if img_entry.get("file_name") == image_name:
                    image_id = img_entry["id"]
                    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == image_id]
                    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
                    results = []
                    for ann in anns:
                        x, y, w, h = ann["bbox"]
                        results.append({
                            "box": [x, y, x + w, y + h],
                            "brand": categories.get(ann.get("category_id"), "UNKNOWN"),
                        })
                    if results:
                        return results
        except Exception:
            pass
    return None


# ── Sidebar Stats ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Status")

    rfdetr_ok = _check_rfdetr_model()
    brand_ok = _check_brand_model()

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


# ── Image Selection ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 1. Select Image")

survey_images = sorted(SURVEY_DIR.glob("*.jpg")) + sorted(SURVEY_DIR.glob("*.jpeg")) + sorted(SURVEY_DIR.glob("*.png")) if SURVEY_DIR.exists() else []

image: Image.Image | None = None
image_source_name: str = ""

col_src1, col_src2 = st.columns(2)

with col_src1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="tester_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name

with col_src2:
    if survey_images:
        img_names = [p.name for p in survey_images]
        selected = st.selectbox("Or select from survey_images/", ["-- Select --"] + img_names, key="tester_select")
        if selected != "-- Select --" and not uploaded:
            img_path = SURVEY_DIR / selected
            image = Image.open(img_path).convert("RGB")
            image_source_name = selected
    else:
        st.info("No images in survey_images/ folder.")

if image is None:
    st.info("Upload an image or select one from survey_images/ to begin testing.")
    st.stop()

st.success(f"**{image_source_name}** -- {image.size[0]}x{image.size[1]}px")


# ── Detection ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2. Run Detection")

if not rfdetr_ok:
    st.warning("No trained model found. Train a model first on the Training page.")
    st.info(f"Expected model at: `{RFDETR_MODEL_PATH}`")
    st.stop()

conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.3, 0.05, key="det_threshold")

if st.button("Run Detection", type="primary", use_container_width=True, key="run_detect"):
    with st.spinner("Running RF-DETR detection..."):
        model = _load_rfdetr()
        if model is None:
            st.error("Failed to load RF-DETR model.")
            st.stop()

        detections, det_time = _run_detection(model, image, threshold=conf_threshold)
        st.session_state["detections"] = detections
        st.session_state["det_time"] = det_time
        st.session_state["det_image_name"] = image_source_name
        # Clear previous classifications
        st.session_state.pop("classifications", None)


# Show detection results
if "detections" in st.session_state and st.session_state.get("det_image_name") == image_source_name:
    detections = st.session_state["detections"]
    det_time = st.session_state.get("det_time", 0)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Packs Detected", len(detections))
    with col_m2:
        if detections:
            avg_conf = np.mean([d["confidence"] for d in detections])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    with col_m3:
        st.metric("Detection Time", f"{det_time:.2f}s")

    if detections:
        # Draw boxes on image
        annotated_img = _draw_boxes(image, detections)
        st.image(annotated_img, caption=f"{len(detections)} detections", use_container_width=True)
    else:
        st.warning("No packs detected. Try lowering the confidence threshold.")
        st.image(image, caption="No detections", use_container_width=True)


    # ── Brand Classification ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. Run Brand Classification")

    if not brand_ok:
        st.warning("Brand classifier not found. Train it first.")
        st.info(f"Expected model at: `{BRAND_MODEL_PATH}`")
    elif not detections:
        st.info("No detections to classify.")
    else:
        if st.button("Classify Brands", type="primary", use_container_width=True, key="run_classify"):
            from brand_classifier import classify_crop
            classifications = []
            progress = st.progress(0, text="Classifying crops...")
            for i, det in enumerate(detections):
                results = classify_crop(det["crop"], top_k=3)
                classifications.append(results)
                progress.progress((i + 1) / len(detections), text=f"Classified {i+1}/{len(detections)}")
            progress.empty()
            st.session_state["classifications"] = classifications

        if "classifications" in st.session_state:
            classifications = st.session_state["classifications"]

            # Re-draw boxes with brand labels
            labels = []
            for cls_list in classifications:
                if cls_list:
                    labels.append(cls_list[0]["brand"])
                else:
                    labels.append("UNKNOWN")

            annotated_img = _draw_boxes(image, detections, labels=labels)
            st.image(annotated_img, caption="Detections with brand labels", use_container_width=True)

            # Show each crop with predictions
            st.markdown("#### Crop Details")
            cols_per_row = 3
            for row_start in range(0, len(detections), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, det_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(detections)))):
                    det = detections[det_idx]
                    cls_list = classifications[det_idx] if det_idx < len(classifications) else []

                    with cols[col_idx]:
                        st.image(det["crop"], caption=f"Crop #{det_idx+1}", width=200)

                        if cls_list:
                            top = cls_list[0]
                            conf_class = "high" if top["confidence"] >= 0.7 else ("medium" if top["confidence"] >= 0.4 else "low")
                            st.markdown(
                                f'<span class="brand-tag">{top["brand"]}</span> '
                                f'<span class="confidence-{conf_class}">{top["confidence"]:.1%}</span>',
                                unsafe_allow_html=True,
                            )
                            # Top-3
                            with st.expander("Top-3 predictions"):
                                for rank, pred in enumerate(cls_list, 1):
                                    st.write(f"{rank}. **{pred['brand']}** -- {pred['confidence']:.1%}")
                        else:
                            st.write("No classification available")


            # ── Side-by-side Comparison ────────────────────────────────────
            ground_truth = _load_ground_truth(image_source_name)

            if ground_truth:
                st.markdown("---")
                st.markdown("### 4. Model vs Ground Truth")

                col_model, col_gt = st.columns(2)

                with col_model:
                    st.markdown("**Model Predictions**")
                    annotated_model = _draw_boxes(image, detections, labels=labels)
                    st.image(annotated_model, use_container_width=True)

                    for i, lbl in enumerate(labels):
                        conf = detections[i]["confidence"]
                        st.write(f"#{i+1}: {lbl} ({conf:.1%})")

                with col_gt:
                    st.markdown("**Ground Truth (Human Annotations)**")
                    gt_labels = [gt.get("brand", "?") for gt in ground_truth]
                    gt_dets = [{"box": gt["box"], "confidence": 1.0} for gt in ground_truth]
                    annotated_gt = _draw_boxes(image, gt_dets, labels=gt_labels)
                    st.image(annotated_gt, use_container_width=True)

                    for i, gt in enumerate(ground_truth):
                        st.write(f"#{i+1}: {gt.get('brand', '?')}")

                # Highlight differences
                model_brands = set(labels)
                gt_brands = set(gt_labels)
                missed = gt_brands - model_brands
                extra = model_brands - gt_brands

                if missed or extra:
                    st.markdown("#### Differences")
                    if missed:
                        for b in missed:
                            st.markdown(f'<span class="diff-removed">Missed: {b}</span>', unsafe_allow_html=True)
                    if extra:
                        for b in extra:
                            st.markdown(f'<span class="diff-added">Extra: {b}</span>', unsafe_allow_html=True)
                else:
                    st.success("Model predictions match ground truth!")


            # ── Quick Correction ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 5. Quick Correction")
            st.caption("If the model got a brand wrong, correct it here. Changes are saved to corrections and training data.")

            correction_made = False
            corrected_brands = []

            for i, det in enumerate(detections):
                cls_list = classifications[i] if i < len(classifications) else []
                current_brand = cls_list[0]["brand"] if cls_list else "UNKNOWN"

                col_crop, col_fix = st.columns([1, 2])
                with col_crop:
                    st.image(det["crop"], width=120, caption=f"Crop #{i+1}")
                with col_fix:
                    st.write(f"Model says: **{current_brand}**")
                    corrected = st.selectbox(
                        f"Correct brand for crop #{i+1}",
                        ["-- Keep --"] + BRAND_LIST,
                        key=f"correct_{i}",
                    )
                    corrected_brands.append(corrected if corrected != "-- Keep --" else current_brand)

            if st.button("Save Corrections", type="primary", use_container_width=True, key="save_corrections"):
                corrections_saved = 0
                for i, det in enumerate(detections):
                    cls_list = classifications[i] if i < len(classifications) else []
                    original = cls_list[0]["brand"] if cls_list else "UNKNOWN"
                    corrected = corrected_brands[i]

                    if corrected != original:
                        save_correction({
                            "serial": image_source_name.replace("serial_", "").split("_")[0],
                            "image_url": image_source_name,
                            "model_used": "rfdetr+brand_classifier",
                            "ai_result": {"brands": [original], "skus": []},
                            "corrected_result": {"brands": [corrected], "skus": []},
                            "notes": json.dumps({
                                "crop_index": i,
                                "bbox": det["box"],
                                "detection_confidence": det["confidence"],
                                "classification_confidence": cls_list[0]["confidence"] if cls_list else 0,
                            }),
                        })
                        corrections_saved += 1

                if corrections_saved > 0:
                    st.success(f"Saved {corrections_saved} correction(s)!")
                else:
                    st.info("No changes to save.")


            # ── Per-Brand Results Table ────────────────────────────────────
            st.markdown("---")
            st.markdown("### 6. Per-Brand Results")

            brand_stats = {}
            for i, lbl in enumerate(labels):
                if lbl not in brand_stats:
                    brand_stats[lbl] = {"count": 0, "confidences": []}
                brand_stats[lbl]["count"] += 1
                if i < len(classifications) and classifications[i]:
                    brand_stats[lbl]["confidences"].append(classifications[i][0]["confidence"])

            table_data = []
            for brand, info in sorted(brand_stats.items()):
                avg_conf = np.mean(info["confidences"]) if info["confidences"] else 0
                table_data.append({
                    "Brand": brand,
                    "Count": info["count"],
                    "Avg Confidence": f"{avg_conf:.1%}",
                })

            if table_data:
                import pandas as pd
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                total_packs = sum(info["count"] for info in brand_stats.values())
                all_confs = [c for info in brand_stats.values() for c in info["confidences"]]
                overall_conf = np.mean(all_confs) if all_confs else 0

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Total Packs", total_packs)
                with col_t2:
                    st.metric("Unique Brands", len(brand_stats))
                with col_t3:
                    st.metric("Processing Time", f"{det_time:.2f}s")

else:
    st.info("Click **Run Detection** to start.")
