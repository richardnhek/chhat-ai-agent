"""
Training — one-click workflow for synthetic data generation, dataset merging,
and RF-DETR training for the CHHAT cigarette pack detection pipeline.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from brands import BRANDS_AND_SKUS
from auth import check_auth

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Training", page_icon="brain", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }

    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 1rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }

    .step-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        margin-bottom: 1rem; border-left: 4px solid #4472C4;
    }
    .step-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.3rem; }
    .step-desc { font-size: 0.9rem; color: #6c757d; }

    .success-banner {
        background: #d4edda; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #28a745; margin-bottom: 1rem;
        color: #155724; font-weight: 500;
    }

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
st.markdown("## Training Pipeline")
st.markdown("Generate synthetic training data, merge datasets, and train the RF-DETR model "
            "for cigarette pack detection.")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
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


# ── Helper functions ──────────────────────────────────────────────────────

def count_files(directory: Path, exts=(".jpg", ".jpeg", ".png")) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def load_coco_stats(path: Path) -> dict:
    """Load basic stats from a COCO annotations file."""
    if not path.exists():
        return {"images": 0, "annotations": 0, "categories": 0}
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
            "categories": len(data.get("categories", [])),
        }
    except Exception:
        return {"images": 0, "annotations": 0, "categories": 0}


def get_model_versions() -> list[dict]:
    """List saved model versions."""
    if not MODELS_DIR.exists():
        return []
    models = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix in (".pt", ".pth", ".onnx"):
            stat = f.stat()
            models.append({
                "name": f.name,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return models


def merge_coco_datasets(real_path: Path, synth_path: Path, output_path: Path) -> dict:
    """Merge real and synthetic COCO annotation files into one unified dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "info": {
            "description": "CHHAT Merged Training Data (Real + Synthetic)",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Build unified category list from both datasets
    all_cat_names = {}  # name -> category dict

    img_id_offset = 0
    ann_id_offset = 0

    for source_label, path in [("real", real_path), ("synthetic", synth_path)]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        # Merge categories
        old_to_new_cat = {}
        for cat in data.get("categories", []):
            name = cat["name"]
            if name not in all_cat_names:
                new_id = len(all_cat_names) + 1
                all_cat_names[name] = {
                    "id": new_id,
                    "name": name,
                    "supercategory": cat.get("supercategory", ""),
                }
            old_to_new_cat[cat["id"]] = all_cat_names[name]["id"]

        # Merge images (remap IDs)
        old_to_new_img = {}
        for img in data.get("images", []):
            new_id = img_id_offset + img["id"]
            old_to_new_img[img["id"]] = new_id
            # Prefix filename with source to avoid collisions
            fname = img["file_name"]
            if source_label == "synthetic" and not fname.startswith("syn_"):
                fname = f"syn_{fname}"
            merged["images"].append({
                **img,
                "id": new_id,
                "file_name": fname,
                "source": source_label,
            })

        # Merge annotations (remap image_id and category_id)
        for ann in data.get("annotations", []):
            new_ann_id = ann_id_offset + ann["id"]
            merged["annotations"].append({
                **ann,
                "id": new_ann_id,
                "image_id": old_to_new_img.get(ann["image_id"], ann["image_id"]),
                "category_id": old_to_new_cat.get(ann["category_id"], ann["category_id"]),
            })

        # Update offsets
        if data.get("images"):
            img_id_offset += max(img["id"] for img in data["images"]) + 1
        if data.get("annotations"):
            ann_id_offset += max(ann["id"] for ann in data["annotations"]) + 1

    merged["categories"] = list(all_cat_names.values())

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return {
        "images": len(merged["images"]),
        "annotations": len(merged["annotations"]),
        "categories": len(merged["categories"]),
    }


# ── Section 1: Current Data Stats ────────────────────────────────────────

st.markdown("---")
st.markdown("### Current Training Data")

col1, col2, col3, col4 = st.columns(4)

# Reference images
ref_count = count_files(REF_DIR)
with col1:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-val">{ref_count}</div>
        <div class="stat-lbl">Reference Pack Images</div>
    </div>""", unsafe_allow_html=True)

# Real annotations
real_stats = load_coco_stats(REAL_COCO)
with col2:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-val">{real_stats['annotations']}</div>
        <div class="stat-lbl">Real Annotations</div>
    </div>""", unsafe_allow_html=True)

# Synthetic data
synth_stats = load_coco_stats(SYNTH_ANN)
synth_img_count = count_files(SYNTH_IMG_DIR)
with col3:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-val">{synth_img_count}</div>
        <div class="stat-lbl">Synthetic Images</div>
    </div>""", unsafe_allow_html=True)

# Merged
merged_stats = load_coco_stats(MERGED_COCO)
with col4:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-val">{merged_stats['annotations']}</div>
        <div class="stat-lbl">Merged Annotations</div>
    </div>""", unsafe_allow_html=True)


# ── Section 2: Generate Synthetic Data ────────────────────────────────────

st.markdown("---")
st.markdown("### Step 1: Generate Synthetic Data")
st.markdown("Composite reference pack images onto shelf backgrounds with augmentations.")

gen_col1, gen_col2 = st.columns([1, 2])

with gen_col1:
    gen_count = st.number_input(
        "Number of images", min_value=10, max_value=50000,
        value=1000, step=100, key="gen_count",
    )
    min_packs = st.slider("Min packs per image", 1, 10, 3, key="min_packs")
    max_packs = st.slider("Max packs per image", 2, 15, 8, key="max_packs")
    if min_packs > max_packs:
        st.warning("Min packs must be <= max packs. Adjusting.")
        min_packs = max_packs

    gen_seed = st.number_input(
        "Random seed (0 = random)", min_value=0, max_value=99999,
        value=0, step=1, key="gen_seed",
    )

with gen_col2:
    if st.button("Generate Synthetic Data", type="primary", use_container_width=True, key="btn_gen"):
        import random as _random
        import numpy as _np
        from synthetic_generator import generate_dataset

        if gen_seed > 0:
            _random.seed(gen_seed)
            _np.random.seed(gen_seed)

        progress_bar = st.progress(0, text="Generating synthetic images...")
        status_text = st.empty()

        def _progress(current, total):
            progress_bar.progress(current / total, text=f"Generating image {current}/{total}...")

        try:
            t0 = time.time()
            ann_path = generate_dataset(
                count=gen_count,
                output_dir=str(SYNTH_DIR),
                packs_per_image=(min_packs, max_packs),
                progress_callback=_progress,
            )
            elapsed = time.time() - t0

            progress_bar.progress(1.0, text="Complete!")
            with open(ann_path) as f:
                result = json.load(f)
            st.markdown(f"""<div class="success-banner">
                Generated {len(result['images'])} images with {len(result['annotations'])} annotations
                in {elapsed:.1f}s
            </div>""", unsafe_allow_html=True)
            st.rerun()

        except Exception as e:
            st.error(f"Generation failed: {e}")

    # Show preview of existing synthetic images
    if SYNTH_IMG_DIR.exists():
        synth_images = sorted(SYNTH_IMG_DIR.glob("*.jpg"))[:6]
        if synth_images:
            st.markdown("**Preview (first 6 images):**")
            preview_cols = st.columns(3)
            for i, img_path in enumerate(synth_images):
                with preview_cols[i % 3]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)


# ── Section 3: Merge Datasets ────────────────────────────────────────────

st.markdown("---")
st.markdown("### Step 2: Merge Training Data")
st.markdown("Combine real annotations (from the annotation tool) and synthetic data "
            "into a single COCO dataset for training.")

merge_col1, merge_col2 = st.columns(2)

with merge_col1:
    st.markdown(f"**Real annotations:** {real_stats['images']} images, {real_stats['annotations']} boxes")
    st.markdown(f"**Synthetic data:** {synth_stats['images']} images, {synth_stats['annotations']} boxes")

with merge_col2:
    if st.button("Merge All Training Data", type="primary", use_container_width=True, key="btn_merge"):
        has_real = REAL_COCO.exists()
        has_synth = SYNTH_ANN.exists()

        if not has_real and not has_synth:
            st.warning("No training data found. Generate synthetic data first or annotate some images.")
        else:
            try:
                result = merge_coco_datasets(REAL_COCO, SYNTH_ANN, MERGED_COCO)

                # Create symlinks to image directories for convenience
                MERGED_IMG_DIR.mkdir(parents=True, exist_ok=True)

                st.markdown(f"""<div class="success-banner">
                    Merged dataset: {result['images']} images, {result['annotations']} annotations,
                    {result['categories']} categories
                </div>""", unsafe_allow_html=True)
                st.rerun()

            except Exception as e:
                st.error(f"Merge failed: {e}")

    if MERGED_COCO.exists():
        st.markdown(f"**Current merged dataset:** {merged_stats['images']} images, "
                    f"{merged_stats['annotations']} annotations, "
                    f"{merged_stats['categories']} categories")


# ── Section 4: Training ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Step 3: Train RF-DETR Model")
st.markdown("Fine-tune the RF-DETR model on your merged training dataset.")

train_col1, train_col2 = st.columns([1, 2])

with train_col1:
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50, step=10, key="epochs")
    batch_size = st.selectbox("Batch size", [2, 4, 8, 16], index=1, key="batch_size")
    lr = st.select_slider("Learning rate", options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3], value=1e-4, key="lr")
    model_size = st.radio("Model size", ["Base", "Large"], index=0, key="model_size")

with train_col2:
    # Determine which dataset to use
    if MERGED_COCO.exists():
        train_dataset = MERGED_COCO
        train_source = "merged"
    elif SYNTH_ANN.exists():
        train_dataset = SYNTH_ANN
        train_source = "synthetic"
    else:
        train_dataset = None
        train_source = None

    if train_dataset:
        ds_stats = load_coco_stats(train_dataset)
        st.info(f"Training on **{train_source}** dataset: "
                f"{ds_stats['images']} images, {ds_stats['annotations']} annotations")

    if st.button("Start RF-DETR Training", type="primary", use_container_width=True, key="btn_train"):
        if not train_dataset or not train_dataset.exists():
            st.warning("No training data available. Generate synthetic data and/or merge first.")
        else:
            # Build the training command
            model_class = "RFDETRBase" if model_size == "Base" else "RFDETRLarge"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model = MODELS_DIR / f"rfdetr_{model_size.lower()}_{timestamp}.pt"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            train_script = f"""
import json
from pathlib import Path

# RF-DETR training
try:
    from rfdetr import {model_class}
    from rfdetr.util.coco_dataset import CocoDetection

    model = {model_class}()

    # The RF-DETR library expects COCO format data
    model.train(
        dataset_dir="{train_dataset.parent}",
        train_ann_file="{train_dataset.name}",
        epochs={epochs},
        batch_size={batch_size},
        lr={lr},
        output_dir="{MODELS_DIR}",
    )
    print("Training complete!")
except ImportError:
    print("ERROR: rfdetr package not installed. Install with: pip install rfdetr")
except Exception as e:
    print(f"Training error: {{e}}")
"""

            st.markdown("**Training command:**")
            cmd = (
                f"python3 -c \""
                f"from rfdetr import {model_class}; "
                f"m = {model_class}(); "
                f"m.train("
                f"dataset_dir='{train_dataset.parent}', "
                f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
                f"output_dir='{MODELS_DIR}')\""
            )
            st.code(cmd, language="bash")

            st.warning(
                "RF-DETR training requires a GPU and the `rfdetr` package. "
                "Copy the command above and run it on a machine with GPU support, "
                "or click below to attempt training on this machine."
            )

            if st.button("Run Training Now", key="btn_run_train"):
                with st.spinner("Training in progress... This may take a while."):
                    try:
                        from rfdetr import RFDETRBase, RFDETRLarge

                        ModelClass = RFDETRBase if model_size == "Base" else RFDETRLarge
                        model = ModelClass()
                        model.train(
                            dataset_dir=str(train_dataset.parent),
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            output_dir=str(MODELS_DIR),
                        )
                        st.success(f"Training complete! Model saved to {MODELS_DIR}")
                        st.rerun()

                    except ImportError:
                        st.error(
                            "The `rfdetr` package is not installed. "
                            "Install it with `pip install rfdetr` and ensure you have GPU support."
                        )
                    except Exception as e:
                        st.error(f"Training failed: {e}")


# ── Section 5: Model Versions ────────────────────────────────────────────

st.markdown("---")
st.markdown("### Model History")

models = get_model_versions()
if models:
    for m in reversed(models):
        st.markdown(
            f"**{m['name']}** — {m['size_mb']:.1f} MB — Last modified: {m['modified']}"
        )
else:
    st.markdown("_No trained models found yet. Train a model to see it here._")


# ── Sidebar summary ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Training Pipeline")
    st.markdown("""
    **Workflow:**
    1. Generate synthetic data from reference pack images
    2. Merge with real annotations from the Annotate tool
    3. Fine-tune RF-DETR on the combined dataset

    **Data sources:**
    - Reference packs: brand book images
    - Real annotations: manually drawn boxes
    - Synthetic: auto-composited shelf scenes
    """)

    st.markdown("---")
    st.markdown(f"Reference images: **{ref_count}**")
    st.markdown(f"Real annotations: **{real_stats['annotations']}**")
    st.markdown(f"Synthetic images: **{synth_img_count}**")
    st.markdown(f"Models trained: **{len(models)}**")
