"""
Annotate — visual annotation/feedback tool for marking missed cigarette boxes.

Draw bounding boxes on images, assign brands + SKUs, and save as training data
in COCO format plus corrections for the self-improving pipeline.
"""

import json
import os
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
import requests
from urllib.parse import urlparse
from PIL import Image
from dotenv import load_dotenv

from streamlit_drawable_canvas import st_canvas

from brands import BRANDS_AND_SKUS
from corrections import save_correction, get_correction_stats
from auth import check_auth

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Annotate", page_icon="🎯", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    .annotation-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem;
        margin-bottom: 0.8rem; border-left: 4px solid #4472C4;
    }
    .brand-tag {
        display: inline-block; background: #4472C4; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 500; margin: 0.15rem;
    }
    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > div { flex: 1 1 100% !important; min-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── CHHAT Branding ────────────────────────────────────────────────────────
st.image("chhat-logo.png", width=120)
st.markdown("## Annotate")
st.markdown("Draw bounding boxes on images to mark cigarette boxes the AI missed. "
            "Annotations are saved as training data and corrections.")

# ── Constants ─────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).parent.parent / "training_data"
ANNOTATIONS_DIR = TRAINING_DIR / "annotations"
IMAGES_DIR = TRAINING_DIR / "images"
COCO_FILE = ANNOTATIONS_DIR / "coco_annotations.json"

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BRAND_LIST = sorted(BRANDS_AND_SKUS.keys())

# Build category mapping (stable IDs for COCO format)
CATEGORY_MAP = {brand: idx + 1 for idx, brand in enumerate(sorted(BRANDS_AND_SKUS.keys()))}


# ── Helpers ───────────────────────────────────────────────────────────────

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


def _get_training_stats() -> tuple[int, int]:
    """Return (num_images, num_boxes) from existing COCO file."""
    coco = _load_coco()
    return len(coco["images"]), len(coco["annotations"])


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_image_bytes(url: str) -> bytes | None:
    """Fetch image from URL with appropriate headers."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "image/*,*/*;q=0.8",
            "Referer": urlparse(url).scheme + "://" + urlparse(url).netloc + "/",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def _load_job_images() -> list[dict]:
    """Load completed job results and extract image URLs."""
    try:
        if os.getenv("SUPABASE_URL"):
            from supabase_db import get_all_jobs
        else:
            from database import get_all_jobs
        jobs = get_all_jobs()
        images = []
        for job in jobs:
            if job.get("status") != "completed":
                continue
            results = job.get("results", [])
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(results, list):
                continue
            for row in results:
                urls = row.get("urls", [])
                serial = row.get("serial", "?")
                for url in urls:
                    if url and isinstance(url, str) and url.startswith("http"):
                        images.append({
                            "url": url,
                            "serial": serial,
                            "job_id": job.get("id", ""),
                            "file_name": job.get("file_name", ""),
                        })
        return images
    except Exception:
        return []


# ── Sidebar: Training Stats ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Training Data")
    n_images, n_boxes = _get_training_stats()
    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{n_images}</div>'
        f'  <div class="stat-lbl">Annotated Images</div>'
        f'</div>'
        f'<div class="stat-box">'
        f'  <div class="stat-val">{n_boxes}</div>'
        f'  <div class="stat-lbl">Bounding Boxes</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Correction stats
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

    if COCO_FILE.exists():
        with open(COCO_FILE, "rb") as f:
            st.download_button(
                "Download COCO Annotations",
                data=f.read(),
                file_name="coco_annotations.json",
                mime="application/json",
                use_container_width=True,
            )


# ── Image Source Selection ────────────────────────────────────────────────

st.markdown("### Select Image")
source_tab = st.radio(
    "Image source",
    ["Upload Image", "From Completed Job"],
    horizontal=True,
    label_visibility="collapsed",
)

image: Image.Image | None = None
image_source_name: str = ""

if source_tab == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image to annotate",
        type=["jpg", "jpeg", "png", "webp"],
        key="annotate_upload",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name

elif source_tab == "From Completed Job":
    job_images = _load_job_images()
    if not job_images:
        st.info("No completed jobs with images found. Upload an image instead, or run an analysis job first.")
    else:
        # Build selection options
        options = [f"[{img['serial']}] {img['file_name']} — {img['url'][:60]}..." for img in job_images]
        selected_idx = st.selectbox(
            "Select an image from completed jobs",
            range(len(options)),
            format_func=lambda i: options[i],
            key="job_image_select",
        )
        if selected_idx is not None:
            img_info = job_images[selected_idx]
            img_bytes = _fetch_image_bytes(img_info["url"])
            if img_bytes:
                import io as _io
                image = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                image_source_name = f"serial_{img_info['serial']}_{img_info['job_id'][:8]}.jpg"
            else:
                st.error("Failed to fetch the image. The URL may have expired.")

if image is None:
    st.stop()

# ── Canvas ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Draw Bounding Boxes")
st.caption("Use the rectangle tool to draw boxes around cigarette packs the AI missed. "
           "You can draw multiple boxes.")

# Scale image to fit canvas while keeping aspect ratio
MAX_CANVAS_WIDTH = 900
img_w, img_h = image.size
scale = min(MAX_CANVAS_WIDTH / img_w, 1.0)
canvas_w = int(img_w * scale)
canvas_h = int(img_h * scale)

display_image = image.resize((canvas_w, canvas_h), Image.LANCZOS)

col_canvas, col_controls = st.columns([3, 1])

with col_controls:
    st.markdown("**Drawing Tools**")
    drawing_mode = st.radio(
        "Mode",
        ["rect", "circle"],
        format_func=lambda x: "Rectangle" if x == "rect" else "Circle",
        key="draw_mode",
    )
    stroke_color = st.color_picker("Box color", "#FF0000", key="stroke_color")
    stroke_width = st.slider("Line width", 1, 6, 3, key="stroke_width")
    fill_opacity = st.slider("Fill opacity", 0, 100, 30, key="fill_opacity")
    fill_color = f"rgba({int(stroke_color[1:3], 16)}, {int(stroke_color[3:5], 16)}, {int(stroke_color[5:7], 16)}, {fill_opacity / 100})"

with col_canvas:
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=display_image,
        drawing_mode=drawing_mode,
        height=canvas_h,
        width=canvas_w,
        key="annotation_canvas",
    )

# ── Process drawn objects ─────────────────────────────────────────────────

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
else:
    objects = []

# Convert canvas objects to bounding boxes (in original image coordinates)
drawn_boxes = []
for i, obj in enumerate(objects):
    if obj.get("type") == "rect":
        # Scale back to original image coordinates
        left = obj["left"] / scale
        top = obj["top"] / scale
        width = (obj["width"] * obj.get("scaleX", 1)) / scale
        height = (obj["height"] * obj.get("scaleY", 1)) / scale
        drawn_boxes.append({
            "index": i,
            "x": round(left),
            "y": round(top),
            "w": round(width),
            "h": round(height),
            "type": "rect",
        })
    elif obj.get("type") == "circle":
        # Convert circle to bounding box
        cx = obj["left"] / scale
        cy = obj["top"] / scale
        r = (obj["radius"] * obj.get("scaleX", 1)) / scale
        drawn_boxes.append({
            "index": i,
            "x": round(cx - r),
            "y": round(cy - r),
            "w": round(2 * r),
            "h": round(2 * r),
            "type": "circle",
        })

if not drawn_boxes:
    st.info("Draw rectangles or circles on the image above to mark missed cigarette boxes.")
    st.stop()

# ── Annotation Form ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Annotations ({len(drawn_boxes)} boxes drawn)")

# Initialize session state for annotations
if "box_annotations" not in st.session_state:
    st.session_state.box_annotations = {}

annotations_valid = True
annotation_data = []

for box in drawn_boxes:
    box_key = f"box_{box['index']}"
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    # Crop thumbnail from original image
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(img_w, x + w)
    crop_y2 = min(img_h, y + h)
    thumbnail = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    st.markdown(
        f'<div class="annotation-card">'
        f'  <strong>Box #{box["index"] + 1}</strong> ({box["type"]}) &mdash; '
        f'  Position: ({x}, {y}), Size: {w} x {h}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_thumb, col_brand, col_sku, col_remove = st.columns([1, 1.5, 2, 0.5])

    with col_thumb:
        st.image(thumbnail, caption=f"Box #{box['index'] + 1}", width=120)

    with col_brand:
        brand = st.selectbox(
            "Mother Brand",
            ["-- Select --"] + BRAND_LIST,
            key=f"brand_{box_key}",
        )

    with col_sku:
        if brand and brand != "-- Select --":
            skus = BRANDS_AND_SKUS.get(brand, [])
            sku = st.selectbox(
                "SKU",
                ["-- Select --"] + skus,
                key=f"sku_{box_key}",
            )
        else:
            sku = st.selectbox("SKU", ["-- Select brand first --"], key=f"sku_{box_key}", disabled=True)

    with col_remove:
        st.markdown("")  # spacer
        # Note: removing objects from canvas requires redrawing; we skip this box on submit instead
        skip = st.checkbox("Skip", key=f"skip_{box_key}", help="Skip this box (don't save)")

    if skip:
        continue

    if brand == "-- Select --" or (brand != "-- Select --" and sku == "-- Select --"):
        annotations_valid = False
    else:
        annotation_data.append({
            "box": box,
            "brand": brand,
            "sku": sku,
        })


# ── Submit ────────────────────────────────────────────────────────────────
st.markdown("---")

n_ready = len(annotation_data)
n_total = len(drawn_boxes)
skipped = sum(1 for box in drawn_boxes if st.session_state.get(f"skip_box_{box['index']}", False))

if n_ready == 0:
    st.warning("Select a brand and SKU for each drawn box before submitting.")
    st.stop()

st.info(f"{n_ready} annotation(s) ready to save.")

if st.button("Submit Annotations", type="primary", use_container_width=True, disabled=(n_ready == 0)):
    # 1) Save the source image to training_data/images/
    img_hash = hashlib.md5(image_source_name.encode()).hexdigest()[:10]
    image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_hash}.jpg"
    image_save_path = IMAGES_DIR / image_filename
    image.save(str(image_save_path), "JPEG", quality=95)

    # 2) Update COCO annotations
    coco = _load_coco()
    image_id = _next_id(coco["images"])

    coco["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": img_w,
        "height": img_h,
    })

    # Ensure categories are up to date
    coco["categories"] = [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()]

    saved_count = 0
    for ann in annotation_data:
        box = ann["box"]
        brand = ann["brand"]
        sku = ann["sku"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        # COCO annotation
        ann_id = _next_id(coco["annotations"])
        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": CATEGORY_MAP.get(brand, 0),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "brand": brand,
            "sku": sku,
        })

        # 3) Save as correction for the self-improving pipeline
        correction = {
            "serial": f"annotate_{image_filename}",
            "image_url": image_source_name,
            "model_used": "human_annotation",
            "ai_result": {"brands": [], "skus": []},
            "corrected_result": {"brands": [brand], "skus": [sku]},
            "notes": f"Manual annotation: bbox=({x},{y},{w},{h})",
        }
        try:
            save_correction(correction)
        except Exception as e:
            st.warning(f"Could not save correction: {e}")

        saved_count += 1

    # Write updated COCO file
    _save_coco(coco)

    st.success(
        f"Saved {saved_count} annotation(s) successfully! "
        f"Image saved to `training_data/images/{image_filename}`. "
        f"COCO annotations updated."
    )
    st.balloons()

    # Refresh sidebar stats
    st.rerun()
