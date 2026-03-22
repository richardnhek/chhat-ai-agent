"""
Outlet Detail View — drill into per-outlet, per-image analysis results from completed jobs.
"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from jobs import get_all_jobs, get_job
from image_analyzer import fetch_image
from enhancements import enhance_image

load_dotenv()

from auth import check_auth

st.set_page_config(page_title="Outlet Detail", page_icon="🔎", layout="wide")

if not check_auth():
    st.stop()

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem; }
    .result-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid #4472C4; }
    .result-card.success { border-left-color: #28a745; }
    .result-card.error { border-left-color: #dc3545; }
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin: 0.15rem; }
    .sku-tag { display: inline-block; background: #6c757d; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; margin: 0.1rem; }
    .metric-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .status-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .status-high { background: #d4edda; color: #155724; }
    .status-medium { background: #fff3cd; color: #856404; }
    .status-low { background: #f8d7da; color: #721c24; }

    /* Enhanced factor bars */
    .factor-bar-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .factor-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .factor-bar-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
    }
    .factor-bar-weight {
        font-size: 0.7rem;
        color: #adb5bd;
        background: #e9ecef;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
    }
    .factor-bar-score {
        font-size: 1rem;
        font-weight: 700;
        min-width: 40px;
        text-align: right;
    }
    .factor-bar-track {
        height: 10px;
        border-radius: 5px;
        background: #e9ecef;
        overflow: hidden;
        position: relative;
    }
    .factor-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    .factor-bar-detail {
        font-size: 0.75rem;
        color: #868e96;
        margin-top: 0.3rem;
    }

    /* Image comparison labels */
    .img-label {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .img-label-original {
        background: #e9ecef;
        color: #495057;
    }
    .img-label-enhanced {
        background: #d4edda;
        color: #155724;
    }

    .image-detail-card { background: #ffffff; border: 1px solid #dee2e6; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem; }
    .notes-box { background: #f1f3f5; border-radius: 8px; padding: 0.8rem; font-size: 0.85rem; color: #495057; margin-top: 0.5rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Mobile responsive — tablet */
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .sub-header { font-size: 0.95rem; margin-bottom: 1rem; }
        .metric-value { font-size: 1.2rem; }
        .brand-tag { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
        .sku-tag { font-size: 0.7rem; padding: 0.15rem 0.4rem; }
        .result-card { padding: 0.8rem; }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }

    /* Mobile responsive — phone */
    @media (max-width: 480px) {
        .main-header { font-size: 1.2rem; }
        .sub-header { font-size: 0.85rem; }
        .metric-value { font-size: 1rem; }
        .brand-tag { font-size: 0.65rem; padding: 0.1rem 0.4rem; }
        .sku-tag { font-size: 0.65rem; padding: 0.1rem 0.3rem; }
        .result-card { padding: 0.6rem; margin-bottom: 0.5rem; }
        [data-testid="stMultiSelect"],
        [data-testid="stDownloadButton"],
        [data-testid="stDownloadButton"] > button {
            width: 100% !important;
        }
        [data-testid="stDownloadButton"] > button {
            min-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── CHHAT Branding ─────────────────────────────────────────────────────
st.image("chhat-logo.png", width=120)
st.markdown('<div class="main-header">Outlet Detail View</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Drill into per-outlet, per-image analysis results from completed jobs.</div>', unsafe_allow_html=True)


# ── Job selection ─────────────────────────────────────────────────────────

jobs = get_all_jobs()
completed_jobs = [j for j in jobs if j["status"] == "completed"]

if not completed_jobs:
    st.info("No completed jobs found. Run an analysis from the Job Manager first.")
    st.stop()

job_options = {
    f"{j['file_name']} — {j['model']} — {j['id']} ({j.get('completed_at', '')[:10]})": j["id"]
    for j in reversed(completed_jobs)
}

selected_label = st.selectbox("Select a completed job", list(job_options.keys()))
selected_job_id = job_options[selected_label]
selected_job = get_job(selected_job_id)

# ── Load results JSON ────────────────────────────────────────────────────

json_path = selected_job.get("results_json_file")
if not json_path:
    # Fall back: try the conventional path
    json_path = f"job_outputs/{selected_job_id}_results.json"

if not Path(json_path).exists():
    st.warning(
        f"No detailed results file found for this job (`{json_path}`). "
        "This job may have been run before the detail-view feature was added. "
        "Re-run the job from the Job Manager to generate detailed results."
    )
    st.stop()

with open(json_path, "r") as f:
    results = json.load(f)

results = [r for r in results if r is not None]

if not results:
    st.warning("Results file is empty.")
    st.stop()

# ── Job summary ──────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Outlets", len(results))
c2.metric("With Brands", len([r for r in results if r.get("brands")]))

all_brands_set = set()
for r in results:
    all_brands_set.update(r.get("brands", []))
c3.metric("Unique Brands", len(all_brands_set))

avg_conf = sum(r.get("confidence_score", 0) for r in results) / len(results) if results else 0
c4.metric("Avg Confidence", f"{avg_conf:.0f}%")

# ── Outlet list ──────────────────────────────────────────────────────────

st.markdown("---")

outlet_labels = []
for r in results:
    serial = r.get("serial", "?")
    brands = r.get("brands", [])
    conf = r.get("confidence", "?")
    conf_score = r.get("confidence_score", "?")
    brand_str = ", ".join(brands) if brands else "No brands"
    outlet_labels.append(f"Serial {serial} — {len(brands)} brands — {conf} ({conf_score}%)")

selected_outlet_idx = st.selectbox(
    "Select an outlet to inspect",
    range(len(outlet_labels)),
    format_func=lambda i: outlet_labels[i],
)

outlet = results[selected_outlet_idx]
serial = outlet.get("serial", "?")
brands = outlet.get("brands", [])
skus = outlet.get("skus", [])
confidence = outlet.get("confidence", "?")
confidence_score = outlet.get("confidence_score", 0)
confidence_factors = outlet.get("confidence_factors", {})
unidentified = outlet.get("unidentified_packs", 0)
per_image = outlet.get("per_image", [])
urls = outlet.get("urls", [])
error = outlet.get("error")

# ── Outlet summary card ──────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"### Outlet: Serial {serial}")

brand_tags = "".join(f'<span class="brand-tag">{b}</span>' for b in brands) if brands else '<span style="color:#999">No cigarette brands detected</span>'
sku_tags = "".join(f'<span class="sku-tag">{s}</span>' for s in skus) if skus else ""

conf_class = f"status-{confidence}"
card_class = "success" if brands else ("error" if error else "success")
unid_html = (
    f'<div><div class="metric-label">Unidentified Packs</div>'
    f'<div style="font-size:1.2rem; font-weight:600; color:#FF8C00;">{unidentified}</div></div>'
    if unidentified else ""
)
error_html = (
    f'<div style="margin-top:0.8rem; color:#dc3545; font-weight:600;">Error: {error}</div>'
    if error else ""
)

st.markdown(
    f'<div class="result-card {card_class}">'
    f'<div style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">'
    f'  <div><div class="metric-label">Serial</div><div style="font-size:1.2rem; font-weight:600;">{serial}</div></div>'
    f'  <div><div class="metric-label">Brand Count</div><div class="metric-value">{len(brands)}</div></div>'
    f'  <div><div class="metric-label">Images</div><div class="metric-value">{len(urls)}</div></div>'
    f'  <div><div class="metric-label">Confidence</div><span class="status-badge {conf_class}">{confidence} ({confidence_score}%)</span></div>'
    f'  {unid_html}'
    f'</div>'
    f'<div style="margin-top:0.8rem;"><div class="metric-label">Brands</div>{brand_tags}</div>'
    + (f'<div style="margin-top:0.5rem;"><div class="metric-label">SKUs</div>{sku_tags}</div>' if sku_tags else "")
    + error_html
    + f'</div>',
    unsafe_allow_html=True,
)

# ── Confidence factor breakdown ──────────────────────────────────────────

if confidence_factors:
    st.markdown("#### Confidence Breakdown")

    factor_display = {
        "ai_confidence": {
            "label": "AI Self-Reported Confidence",
            "weight": "30%",
            "description": "How confident the AI model reported being about its detections.",
        },
        "unidentified_packs": {
            "label": "Unidentified Packs Penalty",
            "weight": "25%",
            "description": "Penalty for cigarette boxes visible but not identifiable.",
        },
        "sku_specificity": {
            "label": "SKU Specificity",
            "weight": "15%",
            "description": "Whether specific SKU variants were identified for each brand.",
        },
        "multi_image_consistency": {
            "label": "Multi-Image Consistency",
            "weight": "20%",
            "description": "Agreement between brands detected across multiple images of the same outlet.",
        },
        "correction_history": {
            "label": "Correction History",
            "weight": "10%",
            "description": "Whether these brands have been frequently corrected in the past.",
        },
    }

    for factor_key, factor_data in confidence_factors.items():
        info = factor_display.get(factor_key, {"label": factor_key, "weight": "?", "description": ""})
        score = factor_data.get("score", 100)
        penalty = factor_data.get("penalty", 0)

        # Determine bar color
        if score >= 80:
            bar_color = "#28a745"
        elif score >= 55:
            bar_color = "#FF8C00"
        else:
            bar_color = "#dc3545"

        # Extra detail for each factor
        detail_parts = []
        if "value" in factor_data:
            detail_parts.append(f"Value: {factor_data['value']}")
        if "count" in factor_data:
            detail_parts.append(f"Count: {factor_data['count']}")
        if "ratio" in factor_data:
            detail_parts.append(f"Ratio: {factor_data['ratio']}")
        if "related_corrections" in factor_data:
            detail_parts.append(f"Related corrections: {factor_data['related_corrections']}")
        if "note" in factor_data:
            detail_parts.append(f"({factor_data['note']})")
        if penalty:
            detail_parts.append(f"Penalty: -{penalty}")
        detail_str = " | ".join(detail_parts) if detail_parts else ""

        st.markdown(
            f'<div class="factor-bar-container">'
            f'  <div class="factor-bar-header">'
            f'    <div>'
            f'      <span class="factor-bar-name">{info["label"]}</span>'
            f'      <span class="factor-bar-weight">Weight: {info["weight"]}</span>'
            f'    </div>'
            f'    <div class="factor-bar-score" style="color:{bar_color};">{score}/100</div>'
            f'  </div>'
            f'  <div class="factor-bar-track">'
            f'    <div class="factor-bar-fill" style="background:{bar_color}; width:{score}%;"></div>'
            f'  </div>'
            + (f'  <div class="factor-bar-detail">{detail_str}</div>' if detail_str else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(info.get("description", "") if len(confidence_factors) == 1 else "Scores are weighted and combined to produce the final confidence score.")

# ── Per-image details ────────────────────────────────────────────────────

if per_image:
    st.markdown("---")
    st.markdown("### Per-Image Analysis")

    for img_idx, img_detail in enumerate(per_image):
        url = img_detail.get("url", "")
        img_error = img_detail.get("error")
        blur_score = img_detail.get("blur_score", 0)
        was_enhanced = img_detail.get("image_enhanced", False)
        img_brands_found = img_detail.get("brands_found", [])
        img_brands = img_detail.get("brands", [])
        img_skus = img_detail.get("skus", [])
        img_confidence = img_detail.get("confidence", "?")
        img_unidentified = img_detail.get("unidentified_packs", 0)
        img_notes = img_detail.get("notes", "")

        st.markdown(f"#### Image {img_idx + 1}")

        if img_error:
            st.error(f"Error processing this image: {img_error}")
            if url:
                st.caption(f"URL: {url}")
            continue

        # ── Image display (original vs enhanced) ─────────────────────
        try:
            image_data, media_type = fetch_image(url)

            if was_enhanced:
                enhanced_data = enhance_image(image_data)

                # Blur score labels for comparison
                if blur_score >= 100:
                    blur_label = "Sharp"
                elif blur_score >= 50:
                    blur_label = "Moderate"
                else:
                    blur_label = "Blurry"

                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.markdown(
                        f'<span class="img-label img-label-original">Original</span>'
                        f' <span style="font-size: 0.75rem; color: #6c757d;">Blur: {blur_score:.1f} ({blur_label})</span>',
                        unsafe_allow_html=True,
                    )
                    st.image(image_data, use_container_width=True)
                with col_enh:
                    st.markdown(
                        f'<span class="img-label img-label-enhanced">Enhanced</span>'
                        f' <span style="font-size: 0.75rem; color: #155724;">Sharpened + denoised</span>',
                        unsafe_allow_html=True,
                    )
                    st.image(enhanced_data, use_container_width=True)
            else:
                st.image(image_data, caption=f"Image {img_idx + 1}", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load image from URL: {e}")
            if url:
                st.caption(f"URL: {url}")

        # ── Image metrics row ────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)

        # Blur score
        if blur_score >= 100:
            blur_label = "Sharp"
            blur_color = "#28a745"
        elif blur_score >= 50:
            blur_label = "Moderate"
            blur_color = "#FF8C00"
        else:
            blur_label = "Blurry"
            blur_color = "#dc3545"

        mc1.markdown(
            f'<div class="metric-label">Blur Score</div>'
            f'<div style="font-size:1.2rem; font-weight:600; color:{blur_color};">{blur_score:.1f} ({blur_label})</div>',
            unsafe_allow_html=True,
        )

        img_conf_class = f"status-{img_confidence}" if img_confidence in ("high", "medium", "low") else ""
        mc2.markdown(
            f'<div class="metric-label">AI Confidence</div>'
            f'<span class="status-badge {img_conf_class}">{img_confidence}</span>',
            unsafe_allow_html=True,
        )

        mc3.markdown(
            f'<div class="metric-label">Brands Detected</div>'
            f'<div style="font-size:1.2rem; font-weight:600;">{len(img_brands)}</div>',
            unsafe_allow_html=True,
        )

        if img_unidentified:
            mc4.markdown(
                f'<div class="metric-label">Unidentified Packs</div>'
                f'<div style="font-size:1.2rem; font-weight:600; color:#FF8C00;">{img_unidentified}</div>',
                unsafe_allow_html=True,
            )
        else:
            mc4.markdown(
                f'<div class="metric-label">Enhanced</div>'
                f'<div style="font-size:1.2rem; font-weight:600;">{"Yes" if was_enhanced else "No"}</div>',
                unsafe_allow_html=True,
            )

        # ── Brand detection details with SKU assignments ─────────────
        if img_brands_found:
            st.markdown("**Brand Detection Details:**")
            for entry in img_brands_found:
                brand = entry.get("brand", "Unknown")
                entry_skus = entry.get("skus", [])
                entry_notes = entry.get("notes", "")

                sku_html = "".join(f'<span class="sku-tag">{s}</span>' for s in entry_skus) if entry_skus else '<span style="color:#adb5bd; font-size:0.8rem;">No specific SKU identified</span>'
                notes_html = f'<div class="notes-box">{entry_notes}</div>' if entry_notes else ""

                st.markdown(
                    f'<div class="image-detail-card">'
                    f'  <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">'
                    f'    <span class="brand-tag">{brand}</span>'
                    f'    <div>{sku_html}</div>'
                    f'  </div>'
                    + notes_html
                    + f'</div>',
                    unsafe_allow_html=True,
                )
        elif not img_error:
            st.info("No cigarette brands detected in this image.")

        # ── AI notes ─────────────────────────────────────────────────
        if img_notes:
            st.markdown(
                f'<div class="notes-box"><strong>AI Notes:</strong> {img_notes}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

elif urls:
    # No per-image data but we have URLs — show basic image display
    st.markdown("---")
    st.markdown("### Outlet Images")
    st.caption("Per-image analysis details are not available for this job. Re-run the job to generate detailed results.")

    img_cols = st.columns(min(len(urls), 3))
    for idx, url in enumerate(urls[:3]):
        try:
            image_data, _ = fetch_image(url)
            with img_cols[idx]:
                st.image(image_data, caption=f"Image {idx + 1}", use_container_width=True)
        except Exception as e:
            with img_cols[idx]:
                st.warning(f"Could not load: {e}")

# ── Raw JSON viewer ──────────────────────────────────────────────────────

with st.expander("View raw JSON data for this outlet", expanded=False):
    display_data = {k: v for k, v in outlet.items() if k != "thumbnails"}
    st.json(display_data)
