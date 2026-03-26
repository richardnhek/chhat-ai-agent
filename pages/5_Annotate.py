"""
Annotate — visual annotation/feedback tool for marking missed cigarette boxes.

Draw bounding boxes on images, assign brands + SKUs, and save as training data
in COCO format plus corrections for the self-improving pipeline.
"""

import json
import io
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


# ── Image Loading — Simple & Direct ──────────────────────────────────────

survey_dir = Path("survey_images")
survey_images = sorted(survey_dir.glob("*.jpg")) if survey_dir.exists() else []

image: Image.Image | None = None
image_source_name: str = ""

if survey_images:
    st.markdown(f"### Select Image ({len(survey_images)} available)")
    img_idx = st.number_input("Image #", min_value=1, max_value=len(survey_images), value=1, step=1, key="img_num") - 1
    img_path = survey_images[img_idx]
    image_source_name = img_path.name

    try:
        image = Image.open(img_path).convert("RGB")
        st.success(f"**{img_path.name}** — {image.size[0]}x{image.size[1]}px")
    except Exception as e:
        st.error(f"Failed to load image: {e}")

    # Also allow upload
    with st.expander("Or upload a different image"):
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], key="ann_upload")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            image_source_name = uploaded.name
else:
    st.warning("No images in survey_images/ folder.")
    uploaded = st.file_uploader("Upload an image to annotate", type=["jpg", "jpeg", "png", "webp"], key="ann_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name

if image is None:
    st.info("No image loaded. Check survey_images/ folder or upload an image.")
    st.stop()

# ── Image Display with Drawing Canvas ─────────────────────────────────────
st.markdown("---")

import base64 as _b64
import streamlit.components.v1 as components

img_w, img_h = image.size
max_display_w = 900
scale = min(max_display_w / img_w, 1.0)
display_w = int(img_w * scale)
display_h = int(img_h * scale)

# Convert image to base64 for embedding in HTML
_buf = io.BytesIO()
image.resize((display_w, display_h), Image.LANCZOS).save(_buf, format="JPEG", quality=85)
img_b64 = _b64.b64encode(_buf.getvalue()).decode()

# Build interactive HTML canvas for drawing rectangles
brand_list_js = json.dumps(sorted(BRANDS_AND_SKUS.keys()))
sku_map_js = json.dumps({k: v for k, v in BRANDS_AND_SKUS.items()})

canvas_html = f"""
<style>
  * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
  .container {{ display: flex; gap: 16px; }}
  .canvas-wrap {{ flex: 1; }}
  .panel {{ width: 320px; max-height: {display_h + 60}px; overflow-y: auto; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; }}
  canvas {{ cursor: crosshair; border: 2px solid #4472C4; border-radius: 6px; display: block; }}
  .box-entry {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin-bottom: 8px; }}
  .box-entry.selected {{ border-color: #4472C4; box-shadow: 0 0 0 2px rgba(68,114,196,0.3); }}
  .box-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
  .box-num {{ font-weight: 700; color: #1a1a2e; }}
  .box-size {{ font-size: 11px; color: #999; }}
  .del-btn {{ background: #ff5252; color: white; border: none; border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 12px; }}
  select, input {{ width: 100%; padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px; margin-bottom: 4px; font-size: 13px; }}
  label {{ font-size: 11px; color: #666; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 2px; }}
  .handle {{ position: absolute; width: 10px; height: 10px; background: white; border: 2px solid #4472C4; cursor: pointer; }}
  .toolbar {{ display: flex; gap: 8px; margin-top: 8px; }}
  .toolbar button {{ padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
  .btn-undo {{ background: #ff9800; color: white; }}
  .btn-clear {{ background: #757575; color: white; }}
  .btn-copy {{ background: #4472C4; color: white; }}
  .count {{ margin-top: 8px; font-size: 13px; color: #333; }}
  #copyResult {{ display: none; margin-top: 6px; padding: 8px; background: #e8f5e9; border-radius: 4px; font-size: 12px; color: #2e7d32; }}
</style>

<div class="container">
  <div class="canvas-wrap">
    <canvas id="c" width="{display_w}" height="{display_h}"></canvas>
    <div class="toolbar">
      <button class="btn-undo" onclick="undo()">Undo</button>
      <button class="btn-clear" onclick="clearAll()">Clear All</button>
      <button class="btn-copy" onclick="copyData()">Copy Annotation Data</button>
    </div>
    <div class="count">Boxes: <strong id="cnt">0</strong></div>
    <div id="copyResult"></div>
  </div>
  <div class="panel" id="panel">
    <div style="font-weight:700;margin-bottom:8px;color:#1a1a2e;">Annotations</div>
    <div id="entries"></div>
  </div>
</div>

<script>
const brands = {brand_list_js};
const skuMap = {sku_map_js};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new Image();
img.src = 'data:image/jpeg;base64,{img_b64}';
const S = {scale};

let boxes = [];
let drawing = false;
let dragging = null; // {{boxIdx, handle}} for resize
let sx, sy;
let selectedBox = -1;

const HANDLE_SIZE = 8;
const HANDLES = ['tl','tr','bl','br','t','b','l','r']; // corners + edges

img.onload = () => redraw(true);

function getHandles(b) {{
  return {{
    tl: [b.x, b.y], tr: [b.x+b.w, b.y],
    bl: [b.x, b.y+b.h], br: [b.x+b.w, b.y+b.h],
    t: [b.x+b.w/2, b.y], b: [b.x+b.w/2, b.y+b.h],
    l: [b.x, b.y+b.h/2], r: [b.x+b.w, b.y+b.h/2],
  }};
}}

function hitHandle(mx, my) {{
  for (let i = boxes.length-1; i >= 0; i--) {{
    const hs = getHandles(boxes[i]);
    for (const [name, [hx,hy]] of Object.entries(hs)) {{
      if (Math.abs(mx-hx) < HANDLE_SIZE && Math.abs(my-hy) < HANDLE_SIZE)
        return {{boxIdx: i, handle: name}};
    }}
  }}
  return null;
}}

let lastPanelCount = -1;

function redrawCanvas() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, {display_w}, {display_h});
  boxes.forEach((b, i) => {{
    const sel = (i === selectedBox);
    ctx.strokeStyle = sel ? '#4472C4' : '#FF0000';
    ctx.lineWidth = sel ? 3 : 2;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = sel ? 'rgba(68,114,196,0.15)' : 'rgba(255,0,0,0.1)';
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = sel ? '#4472C4' : '#FF0000';
    ctx.font = 'bold 13px Arial';
    const label = b.brand || ('#'+(i+1));
    ctx.fillText(label, b.x+3, b.y-4);
    if (sel) {{
      const hs = getHandles(b);
      Object.values(hs).forEach(([hx,hy]) => {{
        ctx.fillStyle = 'white';
        ctx.strokeStyle = '#4472C4';
        ctx.lineWidth = 2;
        ctx.fillRect(hx-4, hy-4, 8, 8);
        ctx.strokeRect(hx-4, hy-4, 8, 8);
      }});
    }}
  }});
  document.getElementById('cnt').textContent = boxes.length;
}}

function redraw(rebuildPanel) {{
  redrawCanvas();
  // Only rebuild panel when boxes are added/removed, NOT during drawing/resizing
  if (rebuildPanel || boxes.length !== lastPanelCount) {{
    lastPanelCount = boxes.length;
    updatePanel();
  }}
}}

function updatePanel() {{
  const el = document.getElementById('entries');
  let html = '';
  boxes.forEach((b, i) => {{
    const rb = {{x:Math.round(b.x/S), y:Math.round(b.y/S), w:Math.round(b.w/S), h:Math.round(b.h/S)}};
    const sel = (i === selectedBox) ? 'selected' : '';
    const brandOpts = brands.map(br => '<option value="'+br+'"'+(b.brand===br?' selected':'')+'>'+br+'</option>').join('');
    const skus = skuMap[b.brand] || [];
    const skuOpts = skus.map(s => '<option value="'+s+'"'+(b.sku===s?' selected':'')+'>'+s+'</option>').join('');
    const typeOpts = ['Individual Pack','Carton (10+ packs)','Sleeve/Bundle','Open Pack'].map(t => '<option'+(b.packType===t?' selected':'')+'>'+t+'</option>').join('');
    const condOpts = ['Clear/Visible','Behind Dirty Glass','Partially Obscured','Upside Down','Sideways','Stacked/Overlapping','Blurry','Reflected/Glare','Far Away/Small'].map(c => '<option'+(b.condition===c?' selected':'')+'>'+c+'</option>').join('');
    const posOpts = ['On Shelf','In Display Case','On Counter','Behind Counter','Hanging','On Floor','On Top of Other Products'].map(p => '<option'+(b.position===p?' selected':'')+'>'+p+'</option>').join('');

    html += '<div class="box-entry '+sel+'" onclick="selectBox('+i+')">';
    html += '<div class="box-header"><span class="box-num">Box #'+(i+1)+'</span><span class="box-size">'+rb.w+'x'+rb.h+'px</span><button class="del-btn" onclick="event.stopPropagation();delBox('+i+')">×</button></div>';
    html += '<label>Brand</label><select onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" onchange="setBoxProp('+i+',\\'brand\\',this.value);updateSkus('+i+',this.value)"><option value="">-- Select --</option>'+brandOpts+'</select>';
    html += '<div id="skuWrap'+i+'"><label>SKU</label><select onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" onchange="setBoxProp('+i+',\\'sku\\',this.value)"><option value="">-- Select --</option>'+skuOpts+'</select></div>';
    html += '<label>Pack Type</label><select onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" onchange="setBoxProp('+i+',\\'packType\\',this.value)"><option value="">-- Select --</option>'+typeOpts+'</select>';
    html += '<label>Condition</label><select onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" onchange="setBoxProp('+i+',\\'condition\\',this.value)"><option value="">-- Select --</option>'+condOpts+'</select>';
    html += '<label>Position</label><select onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" onchange="setBoxProp('+i+',\\'position\\',this.value)"><option value="">-- Select --</option>'+posOpts+'</select>';
    html += '</div>';
  }});
  if (!boxes.length) html = '<div style="color:#999;font-size:13px;text-align:center;padding:20px;">Draw rectangles on the image to start annotating</div>';
  el.innerHTML = html;
}}

function setBoxProp(i, prop, val) {{
  boxes[i][prop] = val;
  if (prop === 'brand') redrawCanvas();
}}

function updateSkus(i, brand) {{
  const skus = skuMap[brand] || [];
  const wrap = document.getElementById('skuWrap'+i);
  if (wrap) {{
    const opts = skus.map(s => '<option value="'+s+'">'+s+'</option>').join('');
    wrap.innerHTML = '<label>SKU</label><select onchange="setBoxProp('+i+',\\'sku\\',this.value)"><option value="">-- Select --</option>'+opts+'</select>';
  }}
}}

function selectBox(i) {{ selectedBox = i; redrawCanvas(); }}
function delBox(i) {{ boxes.splice(i, 1); selectedBox = -1; redraw(true); }}
function undo() {{ boxes.pop(); selectedBox = -1; redraw(true); }}
function clearAll() {{ boxes = []; selectedBox = -1; redraw(true); }}

function copyData() {{
  const data = boxes.map((b,i) => ({{
    box_id: i+1,
    x: Math.round(b.x/S), y: Math.round(b.y/S),
    w: Math.round(b.w/S), h: Math.round(b.h/S),
    brand: b.brand || '', sku: b.sku || '',
    pack_type: b.packType || '', condition: b.condition || '',
    position: b.position || ''
  }}));
  const json = JSON.stringify(data, null, 2);
  navigator.clipboard.writeText(json);
  const el = document.getElementById('copyResult');
  el.style.display = 'block';
  el.textContent = 'Copied ' + data.length + ' annotations to clipboard! Paste into the text area below.';
  setTimeout(() => el.style.display = 'none', 3000);
}}

canvas.addEventListener('mousedown', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  // Check if clicking a handle (resize)
  const hit = hitHandle(mx, my);
  if (hit) {{
    dragging = hit;
    selectedBox = hit.boxIdx;
    sx = mx; sy = my;
    redrawCanvas();
    return;
  }}

  // Check if clicking inside a box (select)
  for (let i = boxes.length-1; i >= 0; i--) {{
    const b = boxes[i];
    if (mx >= b.x && mx <= b.x+b.w && my >= b.y && my <= b.y+b.h) {{
      selectedBox = i;
      redrawCanvas();
      return;
    }}
  }}

  // Start new box
  drawing = true;
  sx = mx; sy = my;
  selectedBox = -1;
}});

canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (dragging) {{
    const b = boxes[dragging.boxIdx];
    const h = dragging.handle;
    if (h.includes('l')) {{ const dx = mx - sx; b.x += dx; b.w -= dx; }}
    if (h.includes('r')) {{ b.w += mx - sx; }}
    if (h.includes('t') && h !== 'tr' || h === 't') {{
      if (h === 't' || h === 'tl') {{ const dy = my - sy; b.y += dy; b.h -= dy; }}
    }}
    if (h === 'tl') {{ const dy = my - sy; b.y += dy; b.h -= dy; }}
    if (h === 'tr') {{ const dy = my - sy; b.y += dy; b.h -= dy; }}
    if (h.includes('b') && h !== 'bl' || h === 'b') {{ b.h += my - sy; }}
    if (h === 'bl') {{ b.h += my - sy; }}
    sx = mx; sy = my;
    redrawCanvas();
    return;
  }}

  if (!drawing) {{
    // Update cursor for handles
    const hit = hitHandle(mx, my);
    canvas.style.cursor = hit ? 'nwse-resize' : 'crosshair';
    return;
  }}

  redrawCanvas();
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.strokeRect(sx, sy, mx-sx, my-sy);
  ctx.setLineDash([]);
}});

canvas.addEventListener('mouseup', (e) => {{
  if (dragging) {{ dragging = null; return; }}
  if (!drawing) return;
  drawing = false;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const w = Math.abs(mx-sx), h = Math.abs(my-sy);
  if (w > 8 && h > 8) {{
    boxes.push({{
      x: Math.min(sx,mx), y: Math.min(sy,my), w, h,
      brand:'', sku:'', packType:'', condition:'', position:''
    }});
    selectedBox = boxes.length - 1;
  }}
  redraw(true);
}});
</script>
"""

st.markdown("### Annotate Cigarette Packs")
st.caption("Draw rectangles around cigarette packs/cartons. Click a box to select it and drag handles to resize. Fill in brand, SKU, and conditions in the right panel.")
components.html(canvas_html, height=display_h + 80)

# Text area to paste annotation data from the canvas
st.markdown("---")
st.markdown("### Save Annotations")
st.caption("Click **Copy Annotation Data** on the canvas above, then paste it below and click Submit.")
pasted_data = st.text_area("Paste annotation data here:", height=100, key=f"paste_{image_source_name}",
                           placeholder='Click "Copy Annotation Data" button above, then paste here...')

if pasted_data and pasted_data.strip().startswith("["):
    try:
        annotations = json.loads(pasted_data)
        brands_found = [a["brand"] for a in annotations if a.get("brand")]
        skus_found = [a["sku"] for a in annotations if a.get("sku")]

        st.success(f"**{len(annotations)} boxes** with {len(set(brands_found))} brands: {', '.join(set(brands_found))}")

        if st.button("Submit Annotations", type="primary", use_container_width=True):
            serial = image_source_name.replace("serial_", "").split("_")[0]

            # Save correction
            save_correction({
                "serial": serial,
                "image_url": image_source_name,
                "model_used": "human_annotation",
                "ai_result": {"brands": [], "skus": []},
                "corrected_result": {"brands": list(set(brands_found)), "skus": list(set(skus_found))},
                "notes": json.dumps([{
                    "brand": a.get("brand"), "sku": a.get("sku"),
                    "pack_type": a.get("pack_type"), "condition": a.get("condition"),
                    "position": a.get("position"), "bbox": [a["x"], a["y"], a["w"], a["h"]]
                } for a in annotations]),
            })

            # Save COCO
            coco = _load_coco()
            image_id = _next_id(coco["images"])
            coco["images"].append({"id": image_id, "file_name": image_source_name, "width": img_w, "height": img_h})

            for a in annotations:
                ann_id = _next_id(coco["annotations"])
                coco["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": 1,
                    "bbox": [a["x"], a["y"], a["w"], a["h"]],
                    "area": a["w"] * a["h"], "iscrowd": 0,
                })

            _save_coco(coco)

            # Save image to training data
            image.save(str(IMAGES_DIR / image_source_name), "JPEG", quality=95)

            st.success(f"Saved {len(annotations)} annotations + image to training data!")
            st.balloons()

    except json.JSONDecodeError:
        st.error("Invalid JSON. Click 'Copy Annotation Data' on the canvas and paste here.")
else:
    st.info("Draw boxes on the image above, fill in brand/SKU/conditions in the right panel, then click **Copy Annotation Data** and paste below.")
