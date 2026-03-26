"""
Training Hub — comprehensive data pipeline management, health checks, model registry,
and one-click training for the CHHAT cigarette pack detection system.
"""

import io
import json
import os
import shutil
import zipfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from brands import BRANDS_AND_SKUS
from auth import check_auth

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Training Hub", page_icon="brain", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & Typography ─────────────────────────── */
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    /* ── Pipeline cards ────────────────────────────── */
    .pipeline-card {
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
        border: 2px solid #e0e0e0; position: relative; min-height: 110px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .pipeline-card.active { border-color: #28a745; background: #d4edda; }
    .pipeline-card.warning { border-color: #ffc107; background: #fff3cd; }
    .pipeline-card.inactive { border-color: #e0e0e0; background: #f8f9fa; }
    .pipeline-val { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .pipeline-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .pipeline-arrow {
        font-size: 1.5rem; color: #adb5bd; display: flex; align-items: center;
        justify-content: center; height: 110px;
    }

    /* ── Stat boxes ────────────────────────────────── */
    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }
    .stat-box.green { border-left-color: #28a745; background: #d4edda; }
    .stat-box.orange { border-left-color: #FF8C00; background: #fff3cd; }
    .stat-box.red { border-left-color: #dc3545; background: #f8d7da; }

    /* ── Brand status badges ───────────────────────── */
    .brand-status { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
    .brand-good { background: #d4edda; color: #155724; }
    .brand-warn { background: #fff3cd; color: #856404; }
    .brand-bad { background: #f8d7da; color: #721c24; }

    /* ── Health check items ─────────────────────────── */
    .health-item {
        background: #f8f9fa; border-radius: 8px; padding: 0.7rem 1rem;
        margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.6rem;
        border-left: 4px solid #e0e0e0;
    }
    .health-item.pass { border-left-color: #28a745; }
    .health-item.warn { border-left-color: #ffc107; }
    .health-item.fail { border-left-color: #dc3545; }
    .health-icon { font-size: 1.1rem; }
    .health-text { font-size: 0.9rem; color: #1a1a2e; }
    .health-detail { font-size: 0.78rem; color: #6c757d; }

    /* ── Model registry ────────────────────────────── */
    .model-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem 1.2rem;
        margin-bottom: 0.8rem; border: 1px solid #e0e0e0;
    }
    .model-card.active-model { border: 2px solid #4472C4; background: #e8f4fd; }
    .model-name { font-size: 1rem; font-weight: 600; color: #1a1a2e; }
    .model-meta { font-size: 0.8rem; color: #6c757d; }

    /* ── Reference gallery ─────────────────────────── */
    .ref-brand-header {
        font-size: 1rem; font-weight: 600; color: #1a1a2e;
        padding: 0.4rem 0; border-bottom: 2px solid #4472C4; margin-bottom: 0.5rem;
    }

    /* ── Command box ───────────────────────────────── */
    .cmd-box {
        background: #1a1a2e; color: #a8e6cf; border-radius: 8px;
        padding: 1rem; font-family: monospace; font-size: 0.85rem;
        overflow-x: auto; margin: 0.5rem 0;
    }

    /* ── Hide Streamlit defaults ───────────────────── */
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
st.markdown('<p class="main-header">Training Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data pipeline management, model training, and evaluation for cigarette pack detection</p>', unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
SURVEY_DIR = BASE / "survey_images"
TRAINING_DIR = BASE / "training_data"
SYNTH_DIR = TRAINING_DIR / "synthetic"
SYNTH_IMG_DIR = SYNTH_DIR / "images"
SYNTH_ANN = SYNTH_DIR / "annotations.json"
REAL_ANN_DIR = TRAINING_DIR / "annotations"
REAL_COCO = REAL_ANN_DIR / "coco_annotations.json"
MERGED_DIR = TRAINING_DIR / "merged"
MERGED_COCO = MERGED_DIR / "annotations.json"
MERGED_IMG_DIR = MERGED_DIR / "images"
MODELS_DIR = BASE / "models"
REF_DIR = BASE / "reference_images"
MAPPING_FILE = REF_DIR / "mapping.json"
CORRECTIONS_DB = BASE / "corrections.json"

# ── Helper functions ──────────────────────────────────────────────────────

def count_files(directory: Path, exts=(".jpg", ".jpeg", ".png")) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def count_files_recursive(directory: Path, exts=(".jpg", ".jpeg", ".png")) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in exts)


def load_coco(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_coco_stats(path: Path) -> dict:
    data = load_coco(path)
    if data is None:
        return {"images": 0, "annotations": 0, "categories": 0}
    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
    }


def get_model_versions() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    models = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix in (".pt", ".pth", ".onnx"):
            stat = f.stat()
            models.append({
                "name": f.name,
                "path": f,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "timestamp": stat.st_mtime,
            })
    return models


def load_mapping() -> dict:
    if not MAPPING_FILE.exists():
        return {}
    try:
        with open(MAPPING_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def load_corrections_count() -> int:
    if not CORRECTIONS_DB.exists():
        return 0
    try:
        with open(CORRECTIONS_DB) as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def get_active_model() -> str | None:
    return st.session_state.get("active_model", None)


def find_all_coco_files() -> list[Path]:
    """Find all COCO annotation files across training_data."""
    coco_files = []
    if TRAINING_DIR.exists():
        for f in TRAINING_DIR.rglob("*.json"):
            if "coco" in f.name.lower() or f.name == "_annotations.coco.json" or f.name == "annotations.json":
                coco_files.append(f)
    return coco_files


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab_overview, tab_health, tab_train, tab_models, tab_import_export, tab_reference = st.tabs([
    "Overview", "Health Check", "Train", "Models", "Import / Export", "Reference"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("### Data Pipeline Overview")
    st.markdown("Visual status of the training data pipeline from raw survey images to trained model.")

    # Gather counts
    raw_count = count_files(SURVEY_DIR)
    real_stats = load_coco_stats(REAL_COCO)
    annotated_count = real_stats["annotations"]
    synth_count = count_files_recursive(SYNTH_DIR)
    merged_stats = load_coco_stats(MERGED_COCO)
    merged_count = merged_stats["images"]
    model_versions = get_model_versions()
    model_count = len(model_versions)
    corrections_count = load_corrections_count()

    # Determine latest model accuracy (from checkpoint if available)
    latest_accuracy = None
    if model_versions:
        # Try to load accuracy from the model checkpoint
        latest_model = max(model_versions, key=lambda m: m["timestamp"])
        try:
            import torch
            checkpoint = torch.load(str(latest_model["path"]), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
                latest_accuracy = checkpoint["val_acc"]
        except Exception:
            pass

    # Pipeline visualization
    cols = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])

    with cols[0]:
        status = "active" if raw_count > 0 else "inactive"
        st.markdown(f'''<div class="pipeline-card {status}">
            <div class="pipeline-val">{raw_count}</div>
            <div class="pipeline-lbl">Raw Images</div>
        </div>''', unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True)

    with cols[2]:
        status = "active" if annotated_count > 0 else ("warning" if raw_count > 0 else "inactive")
        st.markdown(f'''<div class="pipeline-card {status}">
            <div class="pipeline-val">{annotated_count}</div>
            <div class="pipeline-lbl">Annotations</div>
        </div>''', unsafe_allow_html=True)

    with cols[3]:
        st.markdown('<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True)

    with cols[4]:
        status = "active" if synth_count > 0 else "inactive"
        st.markdown(f'''<div class="pipeline-card {status}">
            <div class="pipeline-val">{synth_count}</div>
            <div class="pipeline-lbl">Synthetic</div>
        </div>''', unsafe_allow_html=True)

    with cols[5]:
        st.markdown('<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True)

    with cols[6]:
        status = "active" if model_count > 0 else "inactive"
        v_label = f"v{model_count}" if model_count > 0 else "--"
        st.markdown(f'''<div class="pipeline-card {status}">
            <div class="pipeline-val">{v_label}</div>
            <div class="pipeline-lbl">Trained Models</div>
        </div>''', unsafe_allow_html=True)

    with cols[7]:
        st.markdown('<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True)

    with cols[8]:
        acc_str = f"{latest_accuracy:.0f}%" if latest_accuracy else "--"
        status = "active" if latest_accuracy and latest_accuracy > 80 else ("warning" if latest_accuracy else "inactive")
        st.markdown(f'''<div class="pipeline-card {status}">
            <div class="pipeline-val">{acc_str}</div>
            <div class="pipeline-lbl">Accuracy</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ── Training Data Summary (per-brand stats) ──
    st.markdown("### Training Data Summary")
    st.markdown("Per-brand annotation and reference image counts.")

    mapping = load_mapping()
    coco_data = load_coco(REAL_COCO)

    # Count reference images per mother brand
    ref_per_brand = Counter()
    for filename, info in mapping.items():
        brand_name = info.get("brand", "")
        # Map to mother brand
        mother = None
        brand_upper = brand_name.upper().strip()
        for mb in BRANDS_AND_SKUS:
            if brand_upper.startswith(mb) or mb in brand_upper:
                mother = mb
                break
        if mother:
            ref_per_brand[mother] += 1

    # Count annotations per brand/category from COCO data
    ann_per_brand = Counter()
    if coco_data:
        cat_id_to_name = {}
        for cat in coco_data.get("categories", []):
            cat_id_to_name[cat["id"]] = cat["name"]
        for ann in coco_data.get("annotations", []):
            cat_name = cat_id_to_name.get(ann.get("category_id"), "unknown")
            # Map to mother brand
            cat_upper = cat_name.upper().strip()
            mother = None
            for mb in BRANDS_AND_SKUS:
                if cat_upper.startswith(mb) or mb in cat_upper:
                    mother = mb
                    break
            if mother:
                ann_per_brand[mother] += 1
            else:
                ann_per_brand[cat_name.upper()] += 1

    # Build summary table
    rows = []
    for brand in sorted(BRANDS_AND_SKUS.keys()):
        ref_count_brand = ref_per_brand.get(brand, 0)
        ann_count_brand = ann_per_brand.get(brand, 0)
        total = ref_count_brand + ann_count_brand

        if total >= 20:
            status_html = '<span class="brand-status brand-good">GOOD</span>'
            color = "#d4edda"
        elif total >= 5:
            status_html = '<span class="brand-status brand-warn">LOW</span>'
            color = "#fff3cd"
        else:
            status_html = '<span class="brand-status brand-bad">NEEDS MORE DATA</span>'
            color = "#f8d7da"

        rows.append({
            "Brand": brand,
            "SKU Count": len(BRANDS_AND_SKUS[brand]),
            "Reference Images": ref_count_brand,
            "Annotations": ann_count_brand,
            "Total Examples": total,
            "status_html": status_html,
            "color": color,
        })

    # Render as an HTML table for color-coding
    table_html = """<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
    <thead>
        <tr style="background: #f0f2f6; text-align: left;">
            <th style="padding: 0.5rem 0.8rem;">Brand</th>
            <th style="padding: 0.5rem 0.8rem; text-align: center;">SKUs</th>
            <th style="padding: 0.5rem 0.8rem; text-align: center;">Ref. Images</th>
            <th style="padding: 0.5rem 0.8rem; text-align: center;">Annotations</th>
            <th style="padding: 0.5rem 0.8rem; text-align: center;">Total</th>
            <th style="padding: 0.5rem 0.8rem; text-align: center;">Status</th>
        </tr>
    </thead>
    <tbody>"""

    for r in rows:
        table_html += f"""<tr style="background: {r['color']}22; border-bottom: 1px solid #e9ecef;">
            <td style="padding: 0.4rem 0.8rem; font-weight: 600;">{r['Brand']}</td>
            <td style="padding: 0.4rem 0.8rem; text-align: center;">{r['SKU Count']}</td>
            <td style="padding: 0.4rem 0.8rem; text-align: center;">{r['Reference Images']}</td>
            <td style="padding: 0.4rem 0.8rem; text-align: center;">{r['Annotations']}</td>
            <td style="padding: 0.4rem 0.8rem; text-align: center; font-weight: 600;">{r['Total Examples']}</td>
            <td style="padding: 0.4rem 0.8rem; text-align: center;">{r['status_html']}</td>
        </tr>"""

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Summary stats below table
    st.markdown("")
    scol1, scol2, scol3, scol4 = st.columns(4)
    brands_good = sum(1 for r in rows if r["Total Examples"] >= 20)
    brands_warn = sum(1 for r in rows if 5 <= r["Total Examples"] < 20)
    brands_bad = sum(1 for r in rows if r["Total Examples"] < 5)

    with scol1:
        st.markdown(f'''<div class="stat-box green">
            <div class="stat-val">{brands_good}</div>
            <div class="stat-lbl">Brands with 20+ examples</div>
        </div>''', unsafe_allow_html=True)
    with scol2:
        st.markdown(f'''<div class="stat-box orange">
            <div class="stat-val">{brands_warn}</div>
            <div class="stat-lbl">Brands with 5-19 examples</div>
        </div>''', unsafe_allow_html=True)
    with scol3:
        st.markdown(f'''<div class="stat-box red">
            <div class="stat-val">{brands_bad}</div>
            <div class="stat-lbl">Brands needing more data</div>
        </div>''', unsafe_allow_html=True)
    with scol4:
        st.markdown(f'''<div class="stat-box">
            <div class="stat-val">{corrections_count}</div>
            <div class="stat-lbl">Corrections in database</div>
        </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════

with tab_health:
    st.markdown("### Dataset Health Check")
    st.markdown("Pre-training validation to catch issues before starting a training run.")

    if st.button("Run Health Check", type="primary", key="btn_health"):
        st.session_state["health_check_run"] = True

    if st.session_state.get("health_check_run"):
        checks = []

        # Gather all annotation data
        all_coco_files = find_all_coco_files()
        total_images_all = 0
        total_anns_all = 0
        all_categories = set()
        category_counts = Counter()
        images_with_zero_anns = 0
        all_filenames = []
        anns_per_image = []

        for coco_file in all_coco_files:
            data = load_coco(coco_file)
            if data is None:
                continue

            images = data.get("images", [])
            annotations = data.get("annotations", [])
            categories = data.get("categories", [])

            total_images_all += len(images)
            total_anns_all += len(annotations)

            for cat in categories:
                all_categories.add(cat.get("name", "unknown"))

            # Count annotations per image
            img_ann_count = Counter()
            for ann in annotations:
                img_id = ann.get("image_id")
                img_ann_count[img_id] += 1
                cat_id = ann.get("category_id")
                # Find category name
                cat_name = None
                for c in categories:
                    if c["id"] == cat_id:
                        cat_name = c["name"]
                        break
                if cat_name:
                    category_counts[cat_name] += 1

            for img in images:
                fname = img.get("file_name", "")
                all_filenames.append(fname)
                count = img_ann_count.get(img["id"], 0)
                anns_per_image.append(count)
                if count == 0:
                    images_with_zero_anns += 1

        # Check 1: Total images and annotations
        if total_images_all > 0:
            checks.append(("pass", f"Total images across all datasets: {total_images_all}", ""))
        else:
            checks.append(("fail", "No images found in any dataset", "Generate synthetic data or import annotations"))

        if total_anns_all > 0:
            checks.append(("pass", f"Total annotations: {total_anns_all}", ""))
        else:
            checks.append(("fail", "No annotations found", "Annotate images or generate synthetic data"))

        # Check 2: Average annotations per image
        if anns_per_image:
            avg_anns = sum(anns_per_image) / len(anns_per_image)
            checks.append(("pass", f"Average annotations per image: {avg_anns:.1f}", ""))
        else:
            checks.append(("warn", "Cannot compute average annotations (no data)", ""))

        # Check 3: Images with 0 annotations
        if images_with_zero_anns > 0:
            pct = (images_with_zero_anns / max(total_images_all, 1)) * 100
            if pct > 20:
                checks.append(("fail", f"{images_with_zero_anns} images have 0 annotations ({pct:.0f}%)", "Review and annotate these images or remove them"))
            else:
                checks.append(("warn", f"{images_with_zero_anns} images have 0 annotations ({pct:.0f}%)", "Consider reviewing these"))
        else:
            checks.append(("pass", "All images have at least 1 annotation", ""))

        # Check 4: Duplicate filenames
        filename_counts = Counter(all_filenames)
        duplicates = {f: c for f, c in filename_counts.items() if c > 1}
        if duplicates:
            top_dupes = sorted(duplicates.items(), key=lambda x: -x[1])[:5]
            dupe_str = ", ".join(f"{f} (x{c})" for f, c in top_dupes)
            checks.append(("warn", f"{len(duplicates)} duplicate filenames detected", dupe_str))
        else:
            checks.append(("pass", "No duplicate filenames detected", ""))

        # Check 5: Class count
        checks.append(("pass" if len(all_categories) > 0 else "warn",
                        f"Categories: {len(all_categories)}",
                        ", ".join(sorted(all_categories)[:10]) + ("..." if len(all_categories) > 10 else "")))

        # Check 6: Train/val split
        merged_train_coco = MERGED_DIR / "train" / "_annotations.coco.json"
        merged_val_coco = MERGED_DIR / "valid" / "_annotations.coco.json"
        train_stats = load_coco_stats(merged_train_coco)
        val_stats = load_coco_stats(merged_val_coco)
        if train_stats["images"] > 0 and val_stats["images"] > 0:
            total_split = train_stats["images"] + val_stats["images"]
            val_pct = (val_stats["images"] / total_split) * 100
            status = "pass" if 10 <= val_pct <= 30 else "warn"
            checks.append((status, f"Train/val split: {train_stats['images']} train / {val_stats['images']} val ({val_pct:.0f}% val)", ""))
        elif merged_stats["images"] > 0:
            checks.append(("warn", "Merged dataset exists but no train/val split found", "Run merge to create proper splits"))
        else:
            checks.append(("warn", "No merged dataset with train/val split", "Merge datasets first"))

        # Check 7: Class imbalance
        if category_counts:
            max_count = max(category_counts.values())
            min_count = min(category_counts.values())
            if max_count > 0 and min_count > 0:
                ratio = max_count / min_count
                if ratio > 10:
                    checks.append(("warn", f"Class imbalance detected (max/min ratio: {ratio:.1f}x)", "Consider augmenting under-represented classes"))
                else:
                    checks.append(("pass", f"Class balance acceptable (max/min ratio: {ratio:.1f}x)", ""))

        # Check 8: Reference images
        ref_img_count = count_files(REF_DIR)
        if ref_img_count >= 50:
            checks.append(("pass", f"Reference images: {ref_img_count}", ""))
        elif ref_img_count > 0:
            checks.append(("warn", f"Only {ref_img_count} reference images", "Add more reference images for better classifier training"))
        else:
            checks.append(("fail", "No reference images found", "Add brand reference images to reference_images/"))

        # Render checks
        for status, text, detail in checks:
            icon = {"pass": "&#x2705;", "warn": "&#x26A0;&#xFE0F;", "fail": "&#x274C;"}.get(status, "")
            detail_html = f'<div class="health-detail">{detail}</div>' if detail else ""
            st.markdown(f'''<div class="health-item {status}">
                <div class="health-icon">{icon}</div>
                <div>
                    <div class="health-text">{text}</div>
                    {detail_html}
                </div>
            </div>''', unsafe_allow_html=True)

        # Class distribution chart
        if category_counts and len(category_counts) > 1:
            st.markdown("#### Category Distribution")
            import pandas as pd
            cat_df = pd.DataFrame([
                {"Category": name, "Count": count}
                for name, count in sorted(category_counts.items(), key=lambda x: -x[1])
            ])
            st.bar_chart(cat_df.set_index("Category"))

        # COCO files found
        st.markdown("#### Annotation Files Found")
        for cf in sorted(all_coco_files):
            stats = load_coco_stats(cf)
            rel = cf.relative_to(BASE) if cf.is_relative_to(BASE) else cf
            st.markdown(f"- `{rel}` -- {stats['images']} images, {stats['annotations']} annotations, {stats['categories']} categories")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: TRAIN
# ══════════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown("### One-Click Training Launcher")

    train_section_1, train_section_2 = st.columns(2)

    # ── RF-DETR (Pack Detection) ──
    with train_section_1:
        st.markdown("#### RF-DETR (Pack Detection)")
        st.markdown("Object detection model for finding cigarette packs in shelf images.")

        rfdetr_epochs = st.slider("Epochs", 10, 100, 50, step=10, key="rfdetr_epochs")
        rfdetr_batch = st.selectbox("Batch size", [2, 4, 8, 16], index=0, key="rfdetr_batch")
        rfdetr_lr = st.select_slider(
            "Learning rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            value=1e-4,
            key="rfdetr_lr",
        )
        rfdetr_resolution = st.number_input("Resolution (px)", value=560, step=56, min_value=224, max_value=1120, key="rfdetr_res")
        rfdetr_model_size = st.radio("Model size", ["Base", "Large"], index=0, key="rfdetr_size")

        # Estimated time/cost
        est_time_per_epoch = 3.0 if rfdetr_model_size == "Base" else 6.0  # minutes on A100
        est_total_minutes = est_time_per_epoch * rfdetr_epochs
        est_hours = est_total_minutes / 60
        a100_cost_per_hour = 1.64  # RunPod community A100 80GB
        est_cost = est_hours * a100_cost_per_hour

        st.markdown(f"""
        **Estimated training time:** {est_hours:.1f} hours on A100
        **Estimated RunPod cost:** ${est_cost:.2f} (A100 80GB @ $1.64/hr)
        """)

        model_class = "RFDETRBase" if rfdetr_model_size == "Base" else "RFDETRLarge"

        if st.button("Generate Training Command", type="primary", use_container_width=True, key="btn_rfdetr_cmd"):
            # RunPod command
            runpod_cmd = f"""# RunPod A100 command
pip install rfdetr

python -c "
from rfdetr import {model_class}
model = {model_class}()
model.train(
    dataset_dir='training_data/merged',
    epochs={rfdetr_epochs},
    batch_size={rfdetr_batch},
    lr={rfdetr_lr},
    output_dir='models'
)
"
"""
            st.markdown("**RunPod / GPU command:**")
            st.code(runpod_cmd, language="bash")

            # Local Mac command
            local_cmd = f"""# Local Mac (MPS) command
export PYTORCH_ENABLE_MPS_FALLBACK=1

python train_local.py \\
    --epochs {rfdetr_epochs} \\
    --batch-size {min(rfdetr_batch, 4)} \\
    --lr {rfdetr_lr} \\
    --model-size {rfdetr_model_size.lower()}
"""
            st.markdown("**Local Mac (MPS fallback):**")
            st.code(local_cmd, language="bash")

    # ── Brand Classifier ──
    with train_section_2:
        st.markdown("#### Brand Classifier (ResNet50)")
        st.markdown("Classification model for identifying which brand a detected pack belongs to. Fast enough to train locally.")

        cls_epochs = st.slider("Epochs", 5, 100, 30, step=5, key="cls_epochs")
        cls_batch = st.selectbox("Batch size", [8, 16, 32, 64], index=2, key="cls_batch")
        cls_lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            key="cls_lr",
        )

        ref_img_count = count_files(REF_DIR)
        st.markdown(f"**Reference images available:** {ref_img_count}")
        st.markdown(f"**Training samples per image:** 50 augmentations")
        st.markdown(f"**Estimated total samples:** {ref_img_count * 50}")

        if st.button("Train Brand Classifier Locally", type="primary", use_container_width=True, key="btn_cls_train"):
            if ref_img_count == 0:
                st.warning("No reference images found. Add brand images to reference_images/ first.")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                try:
                    from brand_classifier import train_classifier
                    progress_placeholder.info("Training brand classifier... This typically takes 5-15 minutes on Apple M-series.")

                    output_path = train_classifier(
                        reference_dir=str(REF_DIR),
                        epochs=cls_epochs,
                        batch_size=cls_batch,
                        lr=cls_lr,
                        output_path=str(MODELS_DIR / "brand_classifier.pth"),
                    )

                    progress_placeholder.empty()
                    status_placeholder.success(f"Brand classifier trained successfully! Saved to {output_path}")
                    st.rerun()

                except Exception as e:
                    progress_placeholder.empty()
                    status_placeholder.error(f"Training failed: {e}")

        # Show current classifier info
        cls_model_path = MODELS_DIR / "brand_classifier.pth"
        if cls_model_path.exists():
            stat = cls_model_path.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = stat.st_size / (1024 * 1024)

            try:
                import torch
                checkpoint = torch.load(str(cls_model_path), map_location="cpu", weights_only=False)
                val_acc = checkpoint.get("val_acc", None)
                n_classes = checkpoint.get("num_classes", None)
                acc_str = f" | Val accuracy: {val_acc:.1f}%" if val_acc else ""
                cls_str = f" | Classes: {n_classes}" if n_classes else ""
            except Exception:
                acc_str = ""
                cls_str = ""

            st.markdown(f"""
            **Current classifier:** `brand_classifier.pth`
            - Size: {size_mb:.1f} MB
            - Last trained: {mod_time}{acc_str}{cls_str}
            """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: MODELS
# ══════════════════════════════════════════════════════════════════════════

with tab_models:
    st.markdown("### Model Registry")
    st.markdown("All saved models in the `models/` directory.")

    models = get_model_versions()
    active_model = get_active_model()

    if not models:
        st.info("No models found yet. Train a model from the Train tab to see it here.")
    else:
        for m in sorted(models, key=lambda x: -x["timestamp"]):
            is_active = (active_model == m["name"])
            card_class = "model-card active-model" if is_active else "model-card"
            active_badge = ' <span style="color: #4472C4; font-weight: 600;">[ACTIVE]</span>' if is_active else ""

            # Try to read metrics from model file
            metrics_str = ""
            try:
                import torch
                checkpoint = torch.load(str(m["path"]), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    if "val_acc" in checkpoint:
                        metrics_str += f" | Accuracy: {checkpoint['val_acc']:.1f}%"
                    if "epoch" in checkpoint:
                        metrics_str += f" | Epoch: {checkpoint['epoch']}"
                    if "num_classes" in checkpoint:
                        metrics_str += f" | Classes: {checkpoint['num_classes']}"
            except Exception:
                pass

            st.markdown(f'''<div class="{card_class}">
                <div class="model-name">{m["name"]}{active_badge}</div>
                <div class="model-meta">
                    Size: {m["size_mb"]:.1f} MB &nbsp;|&nbsp; Modified: {m["modified"]}{metrics_str}
                </div>
            </div>''', unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
            with btn_col1:
                if not is_active:
                    if st.button("Set as Active", key=f"activate_{m['name']}", use_container_width=True):
                        st.session_state["active_model"] = m["name"]
                        st.rerun()
                else:
                    st.success("Active", icon=None)
            with btn_col2:
                if st.button("Delete", key=f"delete_{m['name']}", type="secondary", use_container_width=True):
                    try:
                        m["path"].unlink()
                        if active_model == m["name"]:
                            st.session_state["active_model"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: IMPORT / EXPORT
# ══════════════════════════════════════════════════════════════════════════

with tab_import_export:
    import_col, export_col = st.columns(2)

    # ── Import from Roboflow ──
    with import_col:
        st.markdown("### Import from Roboflow")
        st.markdown("Upload a COCO JSON export (zip) from Roboflow to merge into the training data.")

        uploaded_file = st.file_uploader(
            "Upload COCO export (.zip)",
            type=["zip"],
            key="roboflow_upload",
        )

        if uploaded_file:
            try:
                with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
                    file_list = zf.namelist()
                    json_files = [f for f in file_list if f.endswith(".json")]
                    image_files = [f for f in file_list if any(f.lower().endswith(e) for e in (".jpg", ".jpeg", ".png"))]

                    st.markdown(f"**Archive contents:** {len(file_list)} files total")
                    st.markdown(f"- {len(json_files)} JSON annotation file(s)")
                    st.markdown(f"- {len(image_files)} images")

                    # Preview annotations
                    coco_json_file = None
                    for jf in json_files:
                        if "coco" in jf.lower() or "annotation" in jf.lower() or jf.endswith("_annotations.coco.json"):
                            coco_json_file = jf
                            break
                    if not coco_json_file and json_files:
                        coco_json_file = json_files[0]

                    if coco_json_file:
                        preview_data = json.loads(zf.read(coco_json_file))
                        n_imgs = len(preview_data.get("images", []))
                        n_anns = len(preview_data.get("annotations", []))
                        n_cats = len(preview_data.get("categories", []))
                        cat_names = [c["name"] for c in preview_data.get("categories", [])]

                        st.markdown(f"**Preview:** {n_imgs} images, {n_anns} annotations, {n_cats} categories")
                        if cat_names:
                            st.markdown(f"Categories: {', '.join(cat_names[:15])}{'...' if len(cat_names) > 15 else ''}")

                        if st.button("Import into Training Data", type="primary", key="btn_import"):
                            import_dir = TRAINING_DIR / "roboflow_annotated"
                            import_dir.mkdir(parents=True, exist_ok=True)

                            uploaded_file.seek(0)
                            with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf2:
                                zf2.extractall(import_dir)

                            st.success(f"Imported {n_imgs} images and {n_anns} annotations into {import_dir.relative_to(BASE)}")
                            st.info("Run the merge step from the Train tab to include this data in your training set.")
                    else:
                        st.warning("No COCO annotation file found in the archive.")

            except zipfile.BadZipFile:
                st.error("Invalid zip file. Please upload a valid COCO export from Roboflow.")
            except Exception as e:
                st.error(f"Error processing upload: {e}")

    # ── Export Training Data ──
    with export_col:
        st.markdown("### Export Training Data")
        st.markdown("Download the full training dataset as a zip archive.")

        # Show what would be exported
        export_stats = {
            "Real annotations": load_coco_stats(REAL_COCO),
            "Synthetic data": load_coco_stats(SYNTH_ANN),
            "Merged dataset": load_coco_stats(MERGED_COCO),
        }

        for label, stats in export_stats.items():
            if stats["images"] > 0:
                st.markdown(f"- **{label}:** {stats['images']} images, {stats['annotations']} annotations, {stats['categories']} categories")

        if st.button("Export Full Dataset", type="primary", key="btn_export", use_container_width=True):
            # Determine what to export (prefer merged, fallback to individual)
            export_source = None
            if MERGED_COCO.exists():
                export_source = "merged"
            elif REAL_COCO.exists():
                export_source = "real"

            if export_source is None:
                st.warning("No annotation data to export.")
            else:
                with st.spinner("Building export archive..."):
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        if export_source == "merged":
                            # Export merged dataset
                            if MERGED_COCO.exists():
                                zf.write(MERGED_COCO, "annotations.json")
                            # Include train/val splits if they exist
                            for split in ["train", "valid", "test"]:
                                split_dir = MERGED_DIR / split
                                if split_dir.exists():
                                    ann_file = split_dir / "_annotations.coco.json"
                                    if ann_file.exists():
                                        zf.write(ann_file, f"{split}/_annotations.coco.json")
                                    for img_f in split_dir.iterdir():
                                        if img_f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                            zf.write(img_f, f"{split}/{img_f.name}")
                        else:
                            # Export real annotations
                            if REAL_COCO.exists():
                                zf.write(REAL_COCO, "annotations.json")

                    buf.seek(0)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download ZIP",
                        data=buf,
                        file_name=f"chhat_training_data_{timestamp}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 6: REFERENCE GALLERY
# ══════════════════════════════════════════════════════════════════════════

with tab_reference:
    st.markdown("### Quick Reference Gallery")
    st.markdown("All brand reference images organized by brand. Helps annotators quickly identify packs.")

    mapping = load_mapping()

    if not mapping:
        st.info("No reference image mapping found. Add a `mapping.json` to the `reference_images/` directory.")
    else:
        # Organize by mother brand
        brand_images = defaultdict(list)
        for filename, info in sorted(mapping.items()):
            brand_name = info.get("brand", "Unknown")
            # Find mother brand
            mother = None
            brand_upper = brand_name.upper().strip()
            for mb in BRANDS_AND_SKUS:
                if brand_upper.startswith(mb) or mb in brand_upper:
                    mother = mb
                    break
            if not mother:
                mother = brand_name

            img_path = REF_DIR / filename
            if img_path.exists() and img_path.suffix.lower() not in (".wdp",):
                brand_images[mother].append({
                    "filename": filename,
                    "path": img_path,
                    "brand": brand_name,
                    "sku": info.get("sku", brand_name),
                })

        st.markdown(f"**{len(brand_images)} brands** with **{sum(len(v) for v in brand_images.values())} reference images**")
        st.markdown("")

        # Render as collapsible sections
        for brand in sorted(brand_images.keys()):
            images = brand_images[brand]
            sku_count = len(BRANDS_AND_SKUS.get(brand, []))
            with st.expander(f"{brand} ({len(images)} images, {sku_count} SKUs)", expanded=False):
                cols = st.columns(min(len(images), 4))
                for i, img_info in enumerate(images):
                    with cols[i % len(cols)]:
                        try:
                            st.image(
                                str(img_info["path"]),
                                caption=img_info.get("sku", img_info["brand"]),
                                use_container_width=True,
                            )
                        except Exception:
                            st.markdown(f"_Cannot display: {img_info['filename']}_")


# ── Sidebar summary ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Training Hub")
    st.markdown("""
    **Tabs:**
    - **Overview** — Pipeline status and per-brand data
    - **Health Check** — Validate data before training
    - **Train** — Launch RF-DETR or classifier training
    - **Models** — Manage saved model versions
    - **Import/Export** — Roboflow import, dataset export
    - **Reference** — Brand image gallery
    """)

    st.markdown("---")
    st.markdown("**Quick Stats:**")
    st.markdown(f"- Survey images: **{count_files(SURVEY_DIR)}**")
    st.markdown(f"- Reference images: **{count_files(REF_DIR)}**")
    st.markdown(f"- Models: **{len(get_model_versions())}**")
    st.markdown(f"- Corrections: **{load_corrections_count()}**")
