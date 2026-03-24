"""
Review Annotations — Flip through Gemini's auto-annotations, correct errors, save to training data.
"""

import json
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from brands import BRANDS_AND_SKUS
from corrections import save_correction
from auth import check_auth

st.set_page_config(page_title="Review Annotations", page_icon="🔍", layout="wide")

if not check_auth():
    st.stop()

st.image("chhat-logo.png", width=140)
st.markdown("## Review & Correct Annotations")
st.markdown("Flip through each image, verify Gemini's detections, correct mistakes, save to training data.")

# Load Gemini annotations
ann_path = Path("survey_images/gemini_annotations.json")
if not ann_path.exists():
    st.warning("No annotations found. Run Gemini auto-labeling first.")
    st.stop()

with open(ann_path) as f:
    all_annotations = json.load(f)

images_with_packs = [a for a in all_annotations if a["total"] > 0]
images_without = [a for a in all_annotations if a["total"] == 0]

# Stats
c1, c2, c3 = st.columns(3)
c1.metric("Total Images", len(all_annotations))
c2.metric("With Packs", len(images_with_packs))
c3.metric("No Packs", len(images_without))

st.markdown("---")

# Filter
show_filter = st.radio("Show:", ["All images", "Images with packs", "Images without packs"], horizontal=True)
if show_filter == "Images with packs":
    display_list = images_with_packs
elif show_filter == "Images without packs":
    display_list = images_without
else:
    display_list = all_annotations

if not display_list:
    st.info("No images to show with this filter.")
    st.stop()

# Image selector
image_idx = st.slider("Image", 0, len(display_list) - 1, 0, key="img_slider")
ann = display_list[image_idx]

st.markdown(f"### Image {image_idx + 1} of {len(display_list)}: `{ann['image']}`")

# Load and display image with bounding boxes
img_path = Path("survey_images") / ann["image"]
if img_path.exists():
    img = Image.open(img_path)

    # Draw Gemini's detections
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    colors = {
        "MEVIUS": "#1a237e", "WINSTON": "#4a148c", "ESSE": "#0d47a1",
        "FINE": "#b71c1c", "555": "#e65100", "ARA": "#c62828",
        "LUXURY": "#1b5e20", "GOLD SEAL": "#2e7d32", "MARLBORO": "#d32f2f",
        "CAMBO": "#4e342e", "IZA": "#e53935", "HERO": "#ff6f00",
        "COW BOY": "#5d4037", "COCO PALM": "#c62828", "CROWN": "#ffd600",
        "LAPIN": "#7b1fa2", "ORIS": "#00695c",
    }

    for pack in ann.get("packs", []):
        bbox = pack.get("bbox", [])
        brand = pack.get("brand", "UNKNOWN")
        if len(bbox) == 4:
            color = colors.get(brand, "#ff0000")
            draw.rectangle(bbox, outline=color, width=3)
            # Label
            draw.text((bbox[0], bbox[1] - 15), brand, fill=color)

    col_img, col_controls = st.columns([2, 1])

    with col_img:
        st.image(draw_img, caption=f"{ann['image']} — {ann['total']} packs detected", use_container_width=True)

    with col_controls:
        st.markdown(f"**Gemini detected: {ann['total']} packs**")
        st.markdown(f"**Brands:** {', '.join(ann.get('brands', [])) or 'None'}")

        st.markdown("---")
        st.markdown("### Correct This Image")

        # Brand multiselect — pre-filled with Gemini's detections
        all_brand_names = sorted(BRANDS_AND_SKUS.keys())
        gemini_brands = ann.get("brands", [])

        corrected_brands = st.multiselect(
            "Brands present (add/remove):",
            options=all_brand_names,
            default=[b for b in gemini_brands if b in all_brand_names],
            key=f"brands_{image_idx}",
        )

        # SKU selection per brand
        corrected_skus = []
        for brand in corrected_brands:
            brand_skus = BRANDS_AND_SKUS.get(brand, [])
            if brand_skus:
                selected = st.selectbox(
                    f"{brand} SKU:",
                    options=brand_skus,
                    key=f"sku_{brand}_{image_idx}",
                )
                corrected_skus.append(selected)

        # Notes
        notes = st.text_input(
            "Notes:",
            placeholder="e.g., 'Missed 2 packs on top shelf'",
            key=f"notes_{image_idx}",
        )

        # Is this correct?
        is_correct = st.checkbox(
            "Gemini's detection is correct (no changes needed)",
            value=(set(corrected_brands) == set(gemini_brands)),
            key=f"correct_{image_idx}",
        )

        # Save button
        if st.button("Save & Next →", type="primary", use_container_width=True, key=f"save_{image_idx}"):
            # Save correction
            serial = ann["image"].replace("serial_", "").split("_")[0]
            save_correction({
                "serial": serial,
                "image_url": ann["image"],
                "model_used": "gemini-2.5-pro",
                "ai_result": {"brands": gemini_brands, "skus": []},
                "corrected_result": {"brands": corrected_brands, "skus": corrected_skus},
                "notes": notes or ("Confirmed correct" if is_correct else ""),
            })

            # Save to COCO training data
            coco_path = Path("training_data/annotations/coco_annotations.json")
            coco_path.parent.mkdir(parents=True, exist_ok=True)

            if coco_path.exists():
                with open(coco_path) as f:
                    coco = json.load(f)
            else:
                coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cigarette_pack"}]}

            img_id = len(coco["images"]) + 1
            coco["images"].append({
                "id": img_id,
                "file_name": ann["image"],
                "width": ann.get("width", 0),
                "height": ann.get("height", 0),
            })

            for pack in ann.get("packs", []):
                bbox = pack.get("bbox", [])
                if len(bbox) == 4:
                    ann_id = len(coco["annotations"]) + 1
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        "iscrowd": 0,
                    })

            with open(coco_path, "w") as f:
                json.dump(coco, f, indent=2)

            st.success(f"Saved! {len(corrected_brands)} brands, {len(ann.get('packs', []))} boxes → training data")

            # Auto-advance to next image
            if image_idx < len(display_list) - 1:
                st.session_state["img_slider"] = image_idx + 1
                st.rerun()

else:
    st.error(f"Image not found: {img_path}")

# Show progress
st.markdown("---")
reviewed_count = len([1 for a in all_annotations if a["image"] in
    [c.get("image_url", "") for c in []]])  # Would need to check corrections DB
st.caption(f"Tip: Use the slider or 'Save & Next' to flip through images quickly. Each save adds to the training dataset.")
