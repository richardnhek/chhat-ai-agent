"""
Smart Queue — active learning annotation page.

Presents the most valuable images for annotation first, based on model uncertainty,
brand coverage gaps, and outlet diversity. Helps the team focus annotation effort
where it will most improve model accuracy.
"""

import json
import io
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from brands import BRANDS_AND_SKUS
from corrections import save_correction, load_corrections, get_correction_stats
from active_learning import rank_images_for_annotation, get_annotation_stats, get_suggested_brands
from auth import check_auth

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Annotation Queue", page_icon="🧠", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }
    .reason-badge {
        display: inline-block; background: #e8f4fd; color: #1565c0;
        padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;
        margin: 0.15rem; border: 1px solid #bbdefb;
    }
    .score-badge {
        display: inline-block; background: #667eea; color: white;
        padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;
        font-weight: 700;
    }
    .brand-tag {
        display: inline-block; background: #4472C4; color: white;
        padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem;
    }
    .stat-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #4472C4; margin-bottom: 0.8rem;
    }
    .stat-box .stat-val { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .stat-box .stat-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }
    .urgent-brand {
        display: inline-block; background: #fff3cd; color: #856404;
        padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.78rem;
        margin: 0.1rem; border: 1px solid #ffeeba; font-weight: 500;
    }
    .leaderboard-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0; border-bottom: 1px solid #eee;
    }
    .leaderboard-name { font-weight: 500; color: #1a1a2e; }
    .leaderboard-count { font-weight: 700; color: #4472C4; font-size: 1.1rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > div { flex: 1 1 100% !important; min-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── CHHAT Branding ─────────────────────────────────────────────────────────
st.image("chhat-logo.png", width=120)
st.markdown("## Smart Annotation Queue")
st.markdown("Annotate the most valuable images first. The system prioritizes images "
            "where the model is uncertain, brands are under-represented, or new outlets appear.")

# ── Sidebar: Progress & Stats ─────────────────────────────────────────────
ann_stats = get_annotation_stats()
correction_stats = get_correction_stats()
suggested_brands = get_suggested_brands()

with st.sidebar:
    st.markdown("### Annotation Progress")

    total = ann_stats["total_images"]
    annotated = ann_stats["annotated_images"]
    unannotated = ann_stats["unannotated_images"]
    pct = (annotated / total * 100) if total > 0 else 0

    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{annotated} / {total}</div>'
        f'  <div class="stat-lbl">Images Annotated</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(pct / 100, 1.0))
    st.caption(f"{pct:.0f}% complete -- {unannotated} remaining")

    # Brands needing more data
    st.markdown("---")
    st.markdown("### Brands Needing Data")
    brands_needing = ann_stats.get("brands_needing_more", [])
    if brands_needing:
        urgent_html = ""
        for item in brands_needing[:12]:
            urgent_html += f'<span class="urgent-brand">{item["brand"]} ({item["count"]})</span> '
        st.markdown(urgent_html, unsafe_allow_html=True)
        if len(brands_needing) > 12:
            st.caption(f"...and {len(brands_needing) - 12} more")
    else:
        st.success("All brands have sufficient training data.")

    # Today's annotations
    st.markdown("---")
    st.markdown("### Today's Annotations")
    today_str = datetime.now().strftime("%Y-%m-%d")
    all_corrections = load_corrections()
    today_count = 0
    for c in all_corrections:
        ts = c.get("timestamp", "")
        if isinstance(ts, str) and ts.startswith(today_str):
            today_count += 1
    st.markdown(
        f'<div class="stat-box">'
        f'  <div class="stat-val">{today_count}</div>'
        f'  <div class="stat-lbl">Annotations Today</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Leaderboard
    st.markdown("---")
    st.markdown("### Leaderboard")
    team = ann_stats.get("team_stats", {})
    if team:
        sorted_team = sorted(team.items(), key=lambda x: -x[1])
        for name, count in sorted_team:
            display_name = name if name else "unknown"
            st.markdown(
                f'<div class="leaderboard-row">'
                f'  <span class="leaderboard-name">{display_name}</span>'
                f'  <span class="leaderboard-count">{count}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Total corrections: {correction_stats.get('total', 0)}")
    else:
        st.info("No annotations yet.")


# ── Rank images ────────────────────────────────────────────────────────────

ranked = rank_images_for_annotation(top_k=50)

if not ranked:
    st.info("No images available for annotation. Upload survey images to the survey_images/ folder, "
            "or run analysis jobs first.")
    st.stop()

# ── Image selector ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Next Best Image to Annotate")

# Allow scrolling through ranked images
if "sq_index" not in st.session_state:
    st.session_state.sq_index = 0

idx = st.session_state.sq_index
if idx >= len(ranked):
    st.session_state.sq_index = 0
    idx = 0

current = ranked[idx]

# Navigation
nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
with nav_c1:
    if st.button("Previous", use_container_width=True, disabled=idx == 0):
        st.session_state.sq_index = max(0, idx - 1)
        st.rerun()
with nav_c3:
    if st.button("Skip / Next", use_container_width=True, disabled=idx >= len(ranked) - 1):
        st.session_state.sq_index = min(len(ranked) - 1, idx + 1)
        st.rerun()
with nav_c2:
    st.markdown(
        f"<div style='text-align:center; padding: 0.4rem;'>"
        f"Image <strong>{idx + 1}</strong> of <strong>{len(ranked)}</strong> in queue"
        f"</div>",
        unsafe_allow_html=True,
    )

# Show image info
st.markdown("---")

info_c1, info_c2 = st.columns([2, 1])

with info_c2:
    st.markdown(
        f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
        f'border-radius: 12px; padding: 1.2rem; color: white; margin-bottom: 1rem;">'
        f'  <div style="font-size: 0.8rem; text-transform: uppercase; opacity: 0.85; letter-spacing: 0.05em;">Priority Score</div>'
        f'  <div style="font-size: 2.5rem; font-weight: 800; line-height: 1.1;">{current["score"]}</div>'
        f'  <div style="font-size: 0.8rem; opacity: 0.85; margin-top: 0.5rem;">out of 100</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Why this image is valuable:**")
    reasons = current.get("reason", "").split("; ")
    for r in reasons:
        if r.strip():
            st.markdown(f'<span class="reason-badge">{r}</span>', unsafe_allow_html=True)

    if current.get("ai_brands"):
        st.markdown("")
        st.markdown("**AI detected brands:**")
        brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in current["ai_brands"])
        st.markdown(brand_html, unsafe_allow_html=True)

    st.markdown("")
    st.caption(f"Serial: {current.get('serial', 'N/A')}")
    st.caption(f"Confidence: {current.get('confidence', 'N/A')}")

with info_c1:
    # Load and display the image
    image_loaded = False
    pil_image = None

    # Try local path first
    local_path = current.get("local_path", "")
    if local_path and Path(local_path).exists():
        try:
            pil_image = Image.open(local_path).convert("RGB")
            st.image(pil_image, caption=current.get("image", ""), use_container_width=True)
            image_loaded = True
        except Exception:
            pass

    # Try URL
    if not image_loaded and current.get("url"):
        try:
            import requests
            from urllib.parse import urlparse
            url = current["url"]
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "image/*,*/*;q=0.8",
                "Referer": urlparse(url).scheme + "://" + urlparse(url).netloc + "/",
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.image(pil_image, caption=current.get("image", ""), use_container_width=True)
            image_loaded = True
        except Exception as e:
            st.warning(f"Could not load image from URL: {e}")

    if not image_loaded:
        st.warning(f"Could not load image: {current.get('image', 'unknown')}")
        st.caption(f"Path: {local_path or current.get('url', 'N/A')}")


# ── Annotation Form ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Annotate This Image")

BRAND_LIST = sorted(BRANDS_AND_SKUS.keys())

with st.form("annotation_form", clear_on_submit=True):
    # Pre-select AI-detected brands if any
    default_brands = [b for b in current.get("ai_brands", []) if b in BRAND_LIST]

    selected_brands = st.multiselect(
        "Brands visible in this image",
        options=BRAND_LIST,
        default=default_brands,
        help="Select all cigarette brands you can see in the image.",
    )

    # Dynamic SKU selection based on selected brands
    selected_skus = []
    if selected_brands:
        all_skus = []
        for brand in selected_brands:
            all_skus.extend(BRANDS_AND_SKUS.get(brand, []))

        default_skus = [s for s in current.get("ai_skus", []) if s in all_skus]

        selected_skus = st.multiselect(
            "SKUs (specific variants)",
            options=all_skus,
            default=default_skus,
            help="Select the specific SKU variants you can identify.",
        )

    notes = st.text_input("Notes (optional)", placeholder="e.g. blurry image, pack partially hidden")

    col_submit, col_nobrand = st.columns(2)
    with col_submit:
        submitted = st.form_submit_button("Submit Annotation", type="primary", use_container_width=True)
    with col_nobrand:
        no_brands = st.form_submit_button("No Brands Visible", use_container_width=True)

if submitted or no_brands:
    final_brands = [] if no_brands else selected_brands
    final_skus = [] if no_brands else selected_skus
    final_notes = "No cigarette brands visible in image" if no_brands else notes

    # Build correction record
    correction = {
        "serial": current.get("serial", ""),
        "image_url": current.get("image", "") or current.get("url", ""),
        "model_used": "human_annotation",
        "ai_result": {
            "brands": current.get("ai_brands", []),
            "skus": current.get("ai_skus", []),
        },
        "corrected_result": {
            "brands": final_brands,
            "skus": final_skus,
        },
        "notes": final_notes,
    }

    try:
        save_correction(correction)
        st.success(f"Annotation saved! Brands: {', '.join(final_brands) if final_brands else 'none'}")

        # Move to next image
        st.session_state.sq_index = min(idx + 1, len(ranked) - 1)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save annotation: {e}")

# ── Queue Overview ─────────────────────────────────────────────────────────
st.markdown("---")

with st.expander(f"Full Queue ({len(ranked)} images)", expanded=False):
    for i, item in enumerate(ranked):
        score_color = "#28a745" if item["score"] >= 70 else "#FF8C00" if item["score"] >= 40 else "#6c757d"
        brand_tags = "".join(f'<span class="brand-tag">{b}</span>' for b in item.get("ai_brands", [])[:3])
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 12px; padding: 0.5rem 0; '
            f'border-bottom: 1px solid #eee; {"background: #f0f7ff;" if i == idx else ""}">'
            f'  <span style="font-weight: 700; color: {score_color}; min-width: 40px;">{item["score"]}</span>'
            f'  <span style="flex: 1; font-size: 0.85rem; color: #333;">{item["image"][:60]}</span>'
            f'  <span>{brand_tags}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i >= 19:
            st.caption(f"...and {len(ranked) - 20} more")
            break
