"""
Batch Evaluate — Evaluate model accuracy against annotated images and compare models.

Tab 1: Run a model against a holdout set and measure precision, recall, F1, mAP per brand.
Tab 2: Compare two models side-by-side on the same test set.
"""

import io
import os
import json
import time
import zipfile
import tempfile
from pathlib import Path
from collections import defaultdict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

# DINOv2 resolution patch + MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from brands import BRANDS_AND_SKUS
from auth import check_auth

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Batch Evaluate", page_icon="📊", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        text-align: center; border-left: 4px solid #4472C4;
    }
    .metric-card .metric-val { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-card .metric-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }
    .metric-card.green { border-left-color: #28a745; }
    .metric-card.orange { border-left-color: #FF8C00; }
    .metric-card.red { border-left-color: #dc3545; }

    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }

    .f1-green { background: #d4edda; color: #155724; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; }
    .f1-orange { background: #fff3cd; color: #856404; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; }
    .f1-red { background: #f8d7da; color: #721c24; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; }

    .improvement { color: #28a745; font-weight: 700; }
    .regression { color: #dc3545; font-weight: 700; }

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
st.markdown("## Batch Evaluate")
st.markdown("Evaluate model accuracy on annotated test sets and compare models side-by-side.")

# ── Constants ─────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).parent.parent / "training_data"
ANNOTATIONS_DIR = TRAINING_DIR / "annotations"
COCO_FILE = ANNOTATIONS_DIR / "coco_annotations.json"
BRAND_LIST = sorted(BRANDS_AND_SKUS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_model_files() -> list[str]:
    """Find all .pth model files in models/ and output/ directories."""
    root = Path(__file__).parent.parent
    model_files = []
    for search_dir in [root / "models", root / "output"]:
        if search_dir.exists():
            for pth_file in sorted(search_dir.glob("*.pth")):
                model_files.append(str(pth_file.relative_to(root)))
    return model_files


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
def _load_rfdetr(model_path: str):
    """Load an RF-DETR model from a given path."""
    _apply_dinov2_patch()
    try:
        from rfdetr import RFDETRBase
        model = RFDETRBase()
        model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load RF-DETR model: {e}")
        return None


def _load_brand_classifier(model_path: str):
    """Load a brand classifier from the given .pth file."""
    import torch
    from brand_classifier import build_classifier, get_val_transform

    if not Path(model_path).exists():
        return None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["num_classes"]
    model = build_classifier(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    meta = {
        "brand_to_idx": checkpoint["brand_to_idx"],
        "idx_to_brand": checkpoint["idx_to_brand"],
        "val_acc": checkpoint.get("val_acc", 0),
    }
    return model, meta


def _classify_crop_with_model(crop_data: bytes, classifier_model, classifier_meta, top_k: int = 1) -> list[dict]:
    """Classify a crop using a specific classifier model."""
    import torch
    from brand_classifier import get_val_transform

    transform = get_val_transform()
    img = Image.open(io.BytesIO(crop_data)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = classifier_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = probs.topk(top_k)

    results = []
    idx_to_brand = classifier_meta["idx_to_brand"]
    for prob, idx in zip(top_probs, top_indices):
        brand = idx_to_brand.get(str(idx.item()), idx_to_brand.get(idx.item(), "UNKNOWN"))
        results.append({"brand": brand, "confidence": float(prob)})
    return results


def _run_detection(model, pil_image: Image.Image, threshold: float = 0.3) -> list[dict]:
    """Run RF-DETR detection on a PIL image. Returns list of detections."""
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
                })
        return results
    except Exception as e:
        st.warning(f"Detection error: {e}")
        return []


def _load_coco_dataset(coco_path: Path) -> dict:
    """Load COCO annotations and return structured data."""
    with open(coco_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        brand = categories.get(ann.get("category_id"), "UNKNOWN")
        anns_by_image[ann["image_id"]].append({
            "bbox": ann["bbox"],  # [x, y, w, h]
            "brand": brand.upper(),
            "category_id": ann.get("category_id"),
        })

    return {
        "categories": categories,
        "images": images_by_id,
        "annotations_by_image": dict(anns_by_image),
    }


def _iou(box_a, box_b) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _match_detections_to_gt(detections: list[dict], gt_annotations: list[dict], iou_threshold: float = 0.3):
    """
    Match predicted detections to ground truth using IoU.
    Returns: matched_pairs, unmatched_preds (FP), unmatched_gt (FN)
    """
    # Convert gt bbox from [x, y, w, h] to [x1, y1, x2, y2]
    gt_boxes = []
    for ann in gt_annotations:
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x + w, y + h])

    matched_pairs = []  # (pred_idx, gt_idx)
    matched_gt = set()
    matched_pred = set()

    # Greedy matching: highest IoU first
    iou_pairs = []
    for pi, pred in enumerate(detections):
        for gi, gt_box in enumerate(gt_boxes):
            score = _iou(pred["box"], gt_box)
            if score >= iou_threshold:
                iou_pairs.append((score, pi, gi))

    iou_pairs.sort(reverse=True)
    for score, pi, gi in iou_pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pairs.append((pi, gi))
            matched_pred.add(pi)
            matched_gt.add(gi)

    unmatched_preds = [i for i in range(len(detections)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_annotations)) if i not in matched_gt]

    return matched_pairs, unmatched_preds, unmatched_gt


def _compute_metrics(all_results: list[dict]) -> dict:
    """
    Compute precision, recall, F1 per brand and overall from evaluation results.
    Each result: {pred_brands: [...], gt_brands: [...], matched: [(pred_brand, gt_brand)], fp_brands: [...], fn_brands: [...]}
    """
    brand_tp = defaultdict(int)
    brand_fp = defaultdict(int)
    brand_fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[predicted][actual]

    for res in all_results:
        for pred_brand, gt_brand in res.get("matched", []):
            if pred_brand == gt_brand:
                brand_tp[gt_brand] += 1
            else:
                brand_fp[pred_brand] += 1
                brand_fn[gt_brand] += 1
                confusion[pred_brand][gt_brand] += 1

        for brand in res.get("fp_brands", []):
            brand_fp[brand] += 1

        for brand in res.get("fn_brands", []):
            brand_fn[brand] += 1

    all_brands = sorted(set(list(brand_tp.keys()) + list(brand_fp.keys()) + list(brand_fn.keys())))

    per_brand = {}
    for brand in all_brands:
        tp = brand_tp[brand]
        fp = brand_fp[brand]
        fn = brand_fn[brand]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_brand[brand] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    # Overall (micro-average)
    total_tp = sum(brand_tp.values())
    total_fp = sum(brand_fp.values())
    total_fn = sum(brand_fn.values())
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Approximate mAP as mean of per-brand AP (simplified)
    aps = [per_brand[b]["precision"] for b in per_brand if (per_brand[b]["tp"] + per_brand[b]["fn"]) > 0]
    mAP = np.mean(aps) if aps else 0.0

    return {
        "per_brand": per_brand,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "mAP": mAP,
        },
        "confusion": {pred: dict(actuals) for pred, actuals in confusion.items()},
    }


def _evaluate_model(
    rfdetr_model_path: str,
    brand_model_path: str,
    coco_data: dict,
    images_dir: Path,
    num_images: int | None = None,
    det_threshold: float = 0.3,
    progress_callback=None,
) -> tuple[dict, list[dict]]:
    """
    Run evaluation: detect packs + classify brands against ground truth.
    Returns (metrics_dict, per_image_results).
    """
    rfdetr = _load_rfdetr(rfdetr_model_path)
    if rfdetr is None:
        return {}, []

    cls_model, cls_meta = _load_brand_classifier(brand_model_path)
    if cls_model is None:
        st.error(f"Could not load brand classifier: {brand_model_path}")
        return {}, []

    image_ids = list(coco_data["annotations_by_image"].keys())
    if num_images and num_images < len(image_ids):
        import random
        random.seed(42)
        image_ids = random.sample(image_ids, num_images)

    all_results = []
    per_image_details = []

    for idx, img_id in enumerate(image_ids):
        img_info = coco_data["images"].get(img_id)
        if img_info is None:
            continue

        file_name = img_info.get("file_name", "")
        img_path = images_dir / file_name
        if not img_path.exists():
            # Try without subdirectory
            img_path = images_dir.parent / file_name
        if not img_path.exists():
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        gt_annotations = coco_data["annotations_by_image"].get(img_id, [])

        # Detect
        detections = _run_detection(rfdetr, pil_img, threshold=det_threshold)

        # Classify each detection
        pred_brands = []
        for det in detections:
            cls_results = _classify_crop_with_model(det["crop"], cls_model, cls_meta, top_k=1)
            brand = cls_results[0]["brand"] if cls_results else "UNKNOWN"
            pred_brands.append(brand)

        # Match detections to ground truth
        matched_pairs, unmatched_preds, unmatched_gt = _match_detections_to_gt(detections, gt_annotations)

        matched_brand_pairs = []
        for pi, gi in matched_pairs:
            matched_brand_pairs.append((pred_brands[pi], gt_annotations[gi]["brand"]))

        fp_brands = [pred_brands[i] for i in unmatched_preds]
        fn_brands = [gt_annotations[i]["brand"] for i in unmatched_gt]

        # Count errors for this image
        errors = 0
        for pb, gb in matched_brand_pairs:
            if pb != gb:
                errors += 1
        errors += len(fp_brands) + len(fn_brands)

        result = {
            "matched": matched_brand_pairs,
            "fp_brands": fp_brands,
            "fn_brands": fn_brands,
        }
        all_results.append(result)

        per_image_details.append({
            "image_id": img_id,
            "file_name": file_name,
            "errors": errors,
            "pred_brands": pred_brands,
            "gt_brands": [a["brand"] for a in gt_annotations],
            "matched": matched_brand_pairs,
            "fp_brands": fp_brands,
            "fn_brands": fn_brands,
            "pil_image": pil_img,
            "detections": detections,
            "gt_annotations": gt_annotations,
        })

        if progress_callback:
            progress_callback((idx + 1) / len(image_ids), f"Evaluated {idx + 1}/{len(image_ids)} images")

    metrics = _compute_metrics(all_results)
    return metrics, per_image_details


def _draw_boxes(pil_image: Image.Image, boxes: list, labels: list[str] = None, color_default: str = "#4472C4") -> Image.Image:
    """Draw bounding boxes on image."""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    colors = ["#FF4444", "#44AA44", "#4444FF", "#FF8800", "#8844FF", "#FF44AA",
              "#44DDDD", "#AAAA00", "#00AAAA", "#AA44AA"]

    for i, box in enumerate(boxes):
        if isinstance(box, dict):
            coords = box.get("box", box.get("bbox", [0, 0, 0, 0]))
            # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
            if len(coords) == 4 and isinstance(box.get("bbox"), list):
                x, y, w, h = coords
                coords = [x, y, x + w, y + h]
        else:
            coords = box

        x1, y1, x2, y2 = [int(c) for c in coords]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = labels[i] if labels and i < len(labels) else f"#{i+1}"
        text = label
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1), text, fill="white", font=font)

    return img


def _f1_color_class(f1: float) -> str:
    if f1 >= 0.9:
        return "f1-green"
    elif f1 >= 0.7:
        return "f1-orange"
    else:
        return "f1-red"


def _metric_card_color(val: float) -> str:
    if val >= 0.9:
        return "green"
    elif val >= 0.7:
        return "orange"
    else:
        return "red"


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Files")
    model_files = _find_model_files()
    if model_files:
        st.markdown(
            f'<div class="stat-box">'
            f'  <div class="stat-val">{len(model_files)}</div>'
            f'  <div class="stat-lbl">Models Found</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for mf in model_files:
            st.caption(f"  {mf}")
    else:
        st.warning("No .pth model files found in models/ or output/.")

    st.markdown("### Annotations")
    if COCO_FILE.exists():
        try:
            with open(COCO_FILE) as f:
                coco_quick = json.load(f)
            n_images = len(coco_quick.get("images", []))
            n_anns = len(coco_quick.get("annotations", []))
            st.markdown(
                f'<div class="stat-box">'
                f'  <div class="stat-val">{n_images}</div>'
                f'  <div class="stat-lbl">Annotated Images</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="stat-box">'
                f'  <div class="stat-val">{n_anns}</div>'
                f'  <div class="stat-lbl">Total Annotations</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.info("Could not read annotation stats.")
    else:
        st.info("No COCO annotations file found.")


# ── Check for models ─────────────────────────────────────────────────────
model_files = _find_model_files()

if not model_files:
    st.warning("No model files found.")
    st.info(
        "Train a model first on the **Training** page. "
        "Model files (.pth) should be saved in `models/` or `output/` directories.\n\n"
        "Expected locations:\n"
        "- `models/brand_classifier.pth` -- Brand classifier\n"
        "- `output/checkpoint_best_regular.pth` -- RF-DETR detector"
    )
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_eval, tab_compare = st.tabs(["Batch Evaluation", "Model Comparison"])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1: BATCH EVALUATION
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Evaluate Model Accuracy")
    st.caption("Run the full detection + classification pipeline against ground-truth annotations.")

    # ── Model selection ───────────────────────────────────────────────────
    col_det, col_cls = st.columns(2)
    rfdetr_models = [m for m in model_files if "checkpoint" in m.lower() or "rfdetr" in m.lower() or m.startswith("output/")]
    brand_models = [m for m in model_files if "brand" in m.lower() or "classifier" in m.lower() or m.startswith("models/")]

    # Fallback: show all models if no clear match
    if not rfdetr_models:
        rfdetr_models = model_files
    if not brand_models:
        brand_models = model_files

    with col_det:
        selected_rfdetr = st.selectbox("RF-DETR Detector Model", rfdetr_models, key="eval_rfdetr")

    with col_cls:
        selected_brand = st.selectbox("Brand Classifier Model", brand_models, key="eval_brand")

    # ── Test set selection ────────────────────────────────────────────────
    st.markdown("#### Test Set")
    test_source = st.radio(
        "Choose test set source",
        ["Use annotated images (COCO)", "Upload custom holdout set (COCO zip)"],
        key="eval_test_source",
        horizontal=True,
    )

    coco_data = None
    images_dir = None
    num_test_images = None

    if test_source == "Use annotated images (COCO)":
        if not COCO_FILE.exists():
            st.error(f"COCO annotations not found at `{COCO_FILE}`.")
            st.info("Annotate images first using the **Annotate** page, then export as COCO format.")
            st.stop()

        coco_data = _load_coco_dataset(COCO_FILE)
        total_annotated = len(coco_data["annotations_by_image"])

        # Look for images directory
        images_dir = TRAINING_DIR / "images"
        if not images_dir.exists():
            images_dir = TRAINING_DIR

        st.info(f"Found **{total_annotated}** annotated images in the dataset.")

        use_subset = st.checkbox("Use a subset of images", value=False, key="eval_subset")
        if use_subset and total_annotated > 1:
            num_test_images = st.slider(
                "Number of images to evaluate",
                min_value=1,
                max_value=total_annotated,
                value=min(20, total_annotated),
                key="eval_n_images",
            )
        else:
            num_test_images = None

    else:
        uploaded_zip = st.file_uploader(
            "Upload COCO holdout set (.zip)",
            type=["zip"],
            key="eval_upload_zip",
            help="ZIP should contain: annotations.json + an images/ folder",
        )
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), "r") as zf:
                    zf.extractall(tmpdir)

                # Find the annotations file
                ann_files = list(tmpdir.rglob("*.json"))
                if not ann_files:
                    st.error("No JSON annotation file found in the uploaded ZIP.")
                    st.stop()

                ann_file = ann_files[0]
                coco_data = _load_coco_dataset(ann_file)
                images_dir = ann_file.parent / "images"
                if not images_dir.exists():
                    images_dir = ann_file.parent

                # Copy to a persistent temp dir for evaluation
                persist_dir = Path(tempfile.mkdtemp(prefix="chhat_eval_"))
                import shutil
                shutil.copytree(tmpdir, persist_dir, dirs_exist_ok=True)
                images_dir = persist_dir / images_dir.relative_to(tmpdir)

                st.success(f"Loaded {len(coco_data['annotations_by_image'])} annotated images from ZIP.")

    # ── Detection threshold ───────────────────────────────────────────────
    det_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.3, 0.05, key="eval_det_thresh")

    # ── Run Evaluation ────────────────────────────────────────────────────
    if coco_data is not None and st.button("Run Evaluation", type="primary", use_container_width=True, key="run_eval"):
        progress_bar = st.progress(0, text="Starting evaluation...")

        def _update_progress(frac, text):
            progress_bar.progress(frac, text=text)

        start_time = time.time()
        metrics, per_image = _evaluate_model(
            rfdetr_model_path=selected_rfdetr,
            brand_model_path=selected_brand,
            coco_data=coco_data,
            images_dir=images_dir,
            num_images=num_test_images,
            det_threshold=det_threshold,
            progress_callback=_update_progress,
        )
        eval_time = time.time() - start_time
        progress_bar.empty()

        if not metrics:
            st.error("Evaluation failed. Check that models are valid and images exist.")
        else:
            st.session_state["eval_metrics"] = metrics
            st.session_state["eval_per_image"] = per_image
            st.session_state["eval_time"] = eval_time
            st.success(f"Evaluation complete in {eval_time:.1f}s")

    # ── Display results ───────────────────────────────────────────────────
    if "eval_metrics" in st.session_state:
        metrics = st.session_state["eval_metrics"]
        per_image = st.session_state["eval_per_image"]
        eval_time = st.session_state.get("eval_time", 0)
        overall = metrics["overall"]

        # Overall metrics cards
        st.markdown("#### Overall Metrics")
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip(
            [c1, c2, c3, c4],
            [("Precision", overall["precision"]), ("Recall", overall["recall"]),
             ("F1 Score", overall["f1"]), ("mAP", overall["mAP"])],
        ):
            color = _metric_card_color(val)
            with col:
                st.markdown(
                    f'<div class="metric-card {color}">'
                    f'  <div class="metric-val">{val:.1%}</div>'
                    f'  <div class="metric-lbl">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f"*Evaluated {len(per_image)} images in {eval_time:.1f}s*")

        # ── Per-brand table ───────────────────────────────────────────────
        st.markdown("#### Per-Brand Performance")

        per_brand = metrics["per_brand"]
        # Worst-performing brands at top
        sorted_brands = sorted(per_brand.items(), key=lambda x: x[1]["f1"])

        # Highlight worst brands
        worst = [b for b, m in sorted_brands if m["f1"] < 0.7 and (m["tp"] + m["fn"]) > 0]
        if worst:
            st.warning(f"Worst-performing brands: **{', '.join(worst)}**")

        table_rows = []
        for brand, m in sorted_brands:
            f1_cls = _f1_color_class(m["f1"])
            table_rows.append({
                "Brand": brand,
                "TP": m["tp"],
                "FP": m["fp"],
                "FN": m["fn"],
                "Precision": f"{m['precision']:.1%}",
                "Recall": f"{m['recall']:.1%}",
                "F1": m["f1"],
            })

        df_brands = pd.DataFrame(table_rows)

        # Color the F1 column
        def _style_f1(val):
            if isinstance(val, float):
                if val >= 0.9:
                    return "background-color: #d4edda; color: #155724; font-weight: 600"
                elif val >= 0.7:
                    return "background-color: #fff3cd; color: #856404; font-weight: 600"
                else:
                    return "background-color: #f8d7da; color: #721c24; font-weight: 600"
            return ""

        styled_df = df_brands.style.applymap(_style_f1, subset=["F1"]).format({"F1": "{:.1%}"})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ── Confusion Matrix ──────────────────────────────────────────────
        st.markdown("#### Confusion Matrix")
        confusion = metrics.get("confusion", {})

        if confusion:
            # Build matrix
            all_conf_brands = sorted(set(
                list(confusion.keys()) +
                [b for inner in confusion.values() for b in inner.keys()]
            ))

            if all_conf_brands:
                matrix = np.zeros((len(all_conf_brands), len(all_conf_brands)))
                brand_idx_map = {b: i for i, b in enumerate(all_conf_brands)}

                for pred, actuals in confusion.items():
                    for actual, count in actuals.items():
                        pi = brand_idx_map.get(pred, -1)
                        ai = brand_idx_map.get(actual, -1)
                        if pi >= 0 and ai >= 0:
                            matrix[pi][ai] = count

                df_conf = pd.DataFrame(matrix, index=all_conf_brands, columns=all_conf_brands).astype(int)

                # Display as a styled heatmap table
                def _heatmap_style(val):
                    if val == 0:
                        return ""
                    intensity = min(val / max(matrix.max(), 1), 1.0)
                    r = int(255 - intensity * 100)
                    g = int(255 - intensity * 200)
                    b_val = int(255 - intensity * 200)
                    return f"background-color: rgb({r},{g},{b_val}); color: white; font-weight: 600"

                styled_conf = df_conf.style.applymap(_heatmap_style)
                st.caption("Rows = Predicted, Columns = Actual (ground truth)")
                st.dataframe(styled_conf, use_container_width=True)
            else:
                st.info("No brand confusions detected -- all predictions matched ground truth.")
        else:
            st.info("No brand confusions detected -- all predictions matched ground truth.")

        # ── Error images ──────────────────────────────────────────────────
        st.markdown("#### Worst Images (Most Errors)")

        # Sort by error count descending
        error_sorted = sorted(per_image, key=lambda x: x["errors"], reverse=True)
        worst_images = error_sorted[:5]

        if not worst_images or worst_images[0]["errors"] == 0:
            st.success("No errors found across all evaluated images.")
        else:
            for img_detail in worst_images:
                if img_detail["errors"] == 0:
                    continue

                with st.expander(f"{img_detail['file_name']} -- {img_detail['errors']} error(s)", expanded=False):
                    col_pred, col_gt = st.columns(2)

                    with col_pred:
                        st.markdown("**Model Predictions**")
                        pred_img = _draw_boxes(
                            img_detail["pil_image"],
                            img_detail["detections"],
                            labels=img_detail["pred_brands"],
                        )
                        st.image(pred_img, use_container_width=True)
                        for i, b in enumerate(img_detail["pred_brands"]):
                            st.write(f"#{i+1}: {b}")

                    with col_gt:
                        st.markdown("**Ground Truth**")
                        gt_dets = []
                        for ann in img_detail["gt_annotations"]:
                            x, y, w, h = ann["bbox"]
                            gt_dets.append({"box": [x, y, x + w, y + h]})
                        gt_img = _draw_boxes(
                            img_detail["pil_image"],
                            gt_dets,
                            labels=img_detail["gt_brands"],
                        )
                        st.image(gt_img, use_container_width=True)
                        for i, b in enumerate(img_detail["gt_brands"]):
                            st.write(f"#{i+1}: {b}")

                    # Show specific errors
                    if img_detail["fp_brands"]:
                        st.markdown(f"**False positives:** {', '.join(img_detail['fp_brands'])}")
                    if img_detail["fn_brands"]:
                        st.markdown(f"**Missed (false negatives):** {', '.join(img_detail['fn_brands'])}")
                    mismatches = [(p, g) for p, g in img_detail["matched"] if p != g]
                    if mismatches:
                        for p, g in mismatches:
                            st.markdown(f"**Mislabeled:** predicted *{p}*, actual *{g}*")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Compare Two Models")
    st.caption("Evaluate two models on the same holdout set and see which performs better.")

    if len(model_files) < 2:
        st.info(
            "You need at least 2 model files to run a comparison. "
            "Train additional models or place .pth files in `models/` or `output/`."
        )
    else:
        # ── Model A / Model B selection ───────────────────────────────────
        st.markdown("#### Detector Models")
        col_a_det, col_b_det = st.columns(2)
        with col_a_det:
            rfdetr_a = st.selectbox("Model A -- RF-DETR", rfdetr_models, key="cmp_rfdetr_a")
        with col_b_det:
            rfdetr_b = st.selectbox("Model B -- RF-DETR", rfdetr_models, key="cmp_rfdetr_b",
                                    index=min(1, len(rfdetr_models) - 1))

        st.markdown("#### Classifier Models")
        col_a_cls, col_b_cls = st.columns(2)
        with col_a_cls:
            brand_a = st.selectbox("Model A -- Classifier", brand_models, key="cmp_brand_a")
        with col_b_cls:
            brand_b = st.selectbox("Model B -- Classifier", brand_models, key="cmp_brand_b",
                                   index=min(1, len(brand_models) - 1))

        # ── Test set ──────────────────────────────────────────────────────
        if not COCO_FILE.exists():
            st.error("COCO annotations required for comparison. Annotate images first.")
        else:
            cmp_coco = _load_coco_dataset(COCO_FILE)
            cmp_images_dir = TRAINING_DIR / "images"
            if not cmp_images_dir.exists():
                cmp_images_dir = TRAINING_DIR

            total_cmp = len(cmp_coco["annotations_by_image"])
            st.info(f"Comparison will use **{total_cmp}** annotated images.")

            cmp_n_images = st.slider(
                "Number of images for comparison",
                min_value=1,
                max_value=max(total_cmp, 1),
                value=min(20, total_cmp),
                key="cmp_n_images",
            )

            cmp_det_thresh = st.slider("Detection threshold", 0.05, 0.95, 0.3, 0.05, key="cmp_det_thresh")

            if st.button("Run Comparison", type="primary", use_container_width=True, key="run_compare"):
                progress = st.progress(0, text="Evaluating Model A...")

                def _progress_a(frac, text):
                    progress.progress(frac * 0.5, text=f"Model A: {text}")

                def _progress_b(frac, text):
                    progress.progress(0.5 + frac * 0.5, text=f"Model B: {text}")

                metrics_a, per_image_a = _evaluate_model(
                    rfdetr_a, brand_a, cmp_coco, cmp_images_dir,
                    num_images=cmp_n_images, det_threshold=cmp_det_thresh,
                    progress_callback=_progress_a,
                )

                metrics_b, per_image_b = _evaluate_model(
                    rfdetr_b, brand_b, cmp_coco, cmp_images_dir,
                    num_images=cmp_n_images, det_threshold=cmp_det_thresh,
                    progress_callback=_progress_b,
                )

                progress.empty()

                if metrics_a and metrics_b:
                    st.session_state["cmp_metrics_a"] = metrics_a
                    st.session_state["cmp_metrics_b"] = metrics_b
                    st.session_state["cmp_per_image_a"] = per_image_a
                    st.session_state["cmp_per_image_b"] = per_image_b
                    st.session_state["cmp_label_a"] = f"A ({Path(brand_a).stem})"
                    st.session_state["cmp_label_b"] = f"B ({Path(brand_b).stem})"
                    st.success("Comparison complete!")
                else:
                    st.error("One or both evaluations failed. Check model paths.")

            # ── Display comparison ────────────────────────────────────────
            if "cmp_metrics_a" in st.session_state and "cmp_metrics_b" in st.session_state:
                metrics_a = st.session_state["cmp_metrics_a"]
                metrics_b = st.session_state["cmp_metrics_b"]
                per_image_a = st.session_state["cmp_per_image_a"]
                per_image_b = st.session_state["cmp_per_image_b"]
                label_a = st.session_state.get("cmp_label_a", "Model A")
                label_b = st.session_state.get("cmp_label_b", "Model B")

                overall_a = metrics_a["overall"]
                overall_b = metrics_b["overall"]

                # Summary cards
                st.markdown("#### Overall Comparison")
                f1_diff = overall_b["f1"] - overall_a["f1"]
                if f1_diff > 0:
                    st.markdown(
                        f'<div class="metric-card green">'
                        f'  <div class="metric-val">{label_b} is {abs(f1_diff):.1%} better</div>'
                        f'  <div class="metric-lbl">Overall F1 Improvement</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif f1_diff < 0:
                    st.markdown(
                        f'<div class="metric-card red">'
                        f'  <div class="metric-val">{label_b} is {abs(f1_diff):.1%} worse</div>'
                        f'  <div class="metric-lbl">Overall F1 Regression</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'  <div class="metric-val">Identical F1</div>'
                        f'  <div class="metric-lbl">No difference</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Side-by-side overall metrics
                col_oa, col_ob = st.columns(2)
                for col, label, overall in [(col_oa, label_a, overall_a), (col_ob, label_b, overall_b)]:
                    with col:
                        st.markdown(f"**{label}**")
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        for mc, (ml, mv) in zip(
                            [mc1, mc2, mc3, mc4],
                            [("Prec", overall["precision"]), ("Recall", overall["recall"]),
                             ("F1", overall["f1"]), ("mAP", overall["mAP"])],
                        ):
                            with mc:
                                st.metric(ml, f"{mv:.1%}")

                # ── Side-by-side per-brand table ──────────────────────────
                st.markdown("#### Per-Brand Comparison")
                brands_a = metrics_a.get("per_brand", {})
                brands_b = metrics_b.get("per_brand", {})
                all_cmp_brands = sorted(set(list(brands_a.keys()) + list(brands_b.keys())))

                cmp_rows = []
                improvements = []
                regressions = []
                for brand in all_cmp_brands:
                    f1_a = brands_a.get(brand, {}).get("f1", 0.0)
                    f1_b = brands_b.get(brand, {}).get("f1", 0.0)
                    delta = f1_b - f1_a
                    cmp_rows.append({
                        "Brand": brand,
                        f"{label_a} F1": f1_a,
                        f"{label_b} F1": f1_b,
                        "Delta": delta,
                    })
                    if delta > 0.05:
                        improvements.append(brand)
                    elif delta < -0.05:
                        regressions.append(brand)

                df_cmp = pd.DataFrame(cmp_rows)

                def _style_delta(val):
                    if isinstance(val, float):
                        if val > 0.05:
                            return "color: #28a745; font-weight: 700"
                        elif val < -0.05:
                            return "color: #dc3545; font-weight: 700"
                    return ""

                styled_cmp = df_cmp.style.applymap(
                    _style_delta, subset=["Delta"]
                ).format({
                    f"{label_a} F1": "{:.1%}",
                    f"{label_b} F1": "{:.1%}",
                    "Delta": "{:+.1%}",
                })
                st.dataframe(styled_cmp, use_container_width=True, hide_index=True)

                if improvements:
                    st.markdown(f'<span class="improvement">Improved brands: {", ".join(improvements)}</span>', unsafe_allow_html=True)
                if regressions:
                    st.markdown(f'<span class="regression">Regressed brands: {", ".join(regressions)}</span>', unsafe_allow_html=True)

                # ── Image comparison ──────────────────────────────────────
                st.markdown("#### Image Comparison (5 images)")

                # Pair images by file_name
                img_map_a = {d["file_name"]: d for d in per_image_a}
                img_map_b = {d["file_name"]: d for d in per_image_b}
                common_files = sorted(set(img_map_a.keys()) & set(img_map_b.keys()))

                # Pick 5 images with the largest difference in errors
                diff_images = []
                for fname in common_files:
                    da = img_map_a[fname]
                    db = img_map_b[fname]
                    diff_images.append((abs(da["errors"] - db["errors"]), fname))
                diff_images.sort(reverse=True)
                show_files = [f for _, f in diff_images[:5]]

                if not show_files:
                    st.info("No common images to compare.")
                else:
                    for fname in show_files:
                        da = img_map_a[fname]
                        db = img_map_b[fname]

                        with st.expander(
                            f"{fname} -- A: {da['errors']} err, B: {db['errors']} err",
                            expanded=False,
                        ):
                            col_ia, col_ib = st.columns(2)

                            with col_ia:
                                st.markdown(f"**{label_a}** ({da['errors']} errors)")
                                img_a = _draw_boxes(da["pil_image"], da["detections"], labels=da["pred_brands"])
                                st.image(img_a, use_container_width=True)
                                for i, b in enumerate(da["pred_brands"]):
                                    st.write(f"#{i+1}: {b}")

                            with col_ib:
                                st.markdown(f"**{label_b}** ({db['errors']} errors)")
                                img_b = _draw_boxes(db["pil_image"], db["detections"], labels=db["pred_brands"])
                                st.image(img_b, use_container_width=True)
                                for i, b in enumerate(db["pred_brands"]):
                                    st.write(f"#{i+1}: {b}")

                            st.caption(f"Ground truth: {', '.join(da['gt_brands'])}")
