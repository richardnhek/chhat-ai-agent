"""
CHHAT Cigarette Brand Analyzer — Streamlit Dashboard
=====================================================
Upload the CHHAT Excel file -> AI identifies cigarette brands -> Review & Correct -> System improves.
"""

import io
import os
import time
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from image_analyzer import fetch_image, analyze_image, get_available_models
from enhancements import analyze_image_enhanced
from brands import format_q12a, BRANDS_AND_SKUS, BRAND_KHMER
from process import read_raw_data, create_thumbnail, build_output
from corrections import (
    find_relevant_corrections, format_corrections_for_prompt,
    save_correction, get_correction_stats, load_corrections,
)
from confidence import compute_confidence

load_dotenv()

from auth import check_auth

st.set_page_config(
    page_title="CHHAT AI Brand Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not check_auth():
    st.stop()

st.markdown("""
<style>
    /* ── Base & Typography ─────────────────────────── */
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }
    .section-divider { border: none; border-top: 1px solid #e9ecef; margin: 2rem 0; }

    /* ── Result cards ──────────────────────────────── */
    .result-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid #4472C4; }
    .result-card.success { border-left-color: #28a745; }
    .result-card.error { border-left-color: #dc3545; }

    /* ── Tags ──────────────────────────────────────── */
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin: 0.15rem; }
    .sku-tag { display: inline-block; background: #6c757d; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; margin: 0.1rem; }
    .brand-sku-group { display: inline-flex; align-items: center; gap: 3px; margin: 0.15rem 0.1rem; }

    /* ── Metrics ───────────────────────────────────── */
    .metric-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }

    /* ── Status badges ─────────────────────────────── */
    .status-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .status-high { background: #d4edda; color: #155724; }
    .status-medium { background: #fff3cd; color: #856404; }
    .status-low { background: #f8d7da; color: #721c24; }

    /* ── Confidence progress bars ──────────────────── */
    .conf-bar-bg { background: #e9ecef; border-radius: 8px; height: 10px; width: 100%; overflow: hidden; margin-top: 4px; }
    .conf-bar-fill { height: 100%; border-radius: 8px; transition: width 0.5s ease; }
    .conf-bar-fill.high { background: linear-gradient(90deg, #28a745, #34ce57); }
    .conf-bar-fill.medium { background: linear-gradient(90deg, #FF8C00, #ffc107); }
    .conf-bar-fill.low { background: linear-gradient(90deg, #dc3545, #e74c3c); }

    /* ── Correction diffs ──────────────────────────── */
    .correction-added { color: #28a745; font-weight: 600; }
    .correction-removed { color: #dc3545; font-weight: 600; text-decoration: line-through; }
    .diff-box-added { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.9rem; }
    .diff-box-removed { background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.9rem; }
    .diff-box-confirmed { background: #f0f0f0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem 1rem; margin: 0.3rem 0; font-size: 0.85rem; color: #999; }

    /* ── Review summary card ───────────────────────── */
    .review-summary-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.8rem; border: 1px solid #e9ecef;
    }
    .review-serial { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .review-indicator { display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.9rem; font-weight: 600; }
    .indicator-green { color: #28a745; }
    .indicator-orange { color: #FF8C00; }
    .indicator-red { color: #dc3545; }

    /* ── Footer ────────────────────────────────────── */
    .app-footer {
        text-align: center; color: #adb5bd; font-size: 0.8rem;
        padding: 2rem 0 1rem 0; border-top: 1px solid #e9ecef; margin-top: 3rem;
    }

    /* ── Download cards ────────────────────────────── */
    .download-card {
        background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .download-card h4 { margin: 0.3rem 0; font-size: 1rem; color: #1a1a2e; }
    .download-card p { margin: 0; font-size: 0.8rem; color: #6c757d; }

    /* ── Hide defaults ─────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Mobile responsive — tablet ────────────────── */
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .sub-header { font-size: 0.95rem; margin-bottom: 1rem; }
        .metric-value { font-size: 1.2rem; }
        .brand-tag { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
        .sku-tag { font-size: 0.7rem; padding: 0.15rem 0.4rem; }
        .result-card { padding: 0.8rem; }
        .result-card > div[style*="display:flex"] {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        [data-testid="stSidebar"] div[style*="font-size:0.85rem"] {
            font-size: 0.75rem !important;
            line-height: 1.4 !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }

    /* ── Mobile responsive — phone ─────────────────── */
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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    all_models = get_available_models()
    model = st.selectbox("AI Model", all_models, index=0,
        help="Claude = Anthropic, Gemini = Google, Qwen = Open-source (via Fireworks)")
    delay = st.slider("Delay between calls (sec)", 0.5, 5.0, 1.5, 0.5)
    photo_cols = st.text_input("Q32 Photo columns", value="B,C,D", help="Comma-separated column letters")
    start_row = st.number_input("Data starts at row", value=3, min_value=2)

    st.markdown("---")
    st.markdown("### Enhancements")
    enable_enhancement = st.checkbox("Image Enhancement", value=True,
        help="Sharpening, denoising, contrast boost, glare reduction")
    enable_ocr = st.checkbox("OCR Text Pre-scan", value=True,
        help="Extract text from packs first to help identify brands/SKUs")
    enable_sku_refinement = st.checkbox("Two-Pass SKU Refinement", value=True,
        help="Second focused pass to identify exact SKU variants")
    enable_cross_val = st.checkbox("Cross-Model Validation", value=False,
        help="Run through 2 models and compare (doubles API cost)")
    cross_val_models = None
    if enable_cross_val:
        cross_val_models = st.multiselect("Validation models",
            all_models, default=[all_models[0], all_models[5]] if len(all_models) > 5 else all_models[:2],
            key="cross_val_models")

    st.markdown("---")
    stats = get_correction_stats()
    st.markdown("### Learning Stats")
    st.metric("Past Corrections", stats["total"])
    if stats["total"] > 0:
        st.caption(
            f"Brand swaps: {stats['brand_swaps']} | "
            f"Added: {stats['brand_added']} | "
            f"Removed: {stats['brand_removed']} | "
            f"SKU fixes: {stats['sku_corrected']}"
        )
        st.success("System is learning from corrections")
    else:
        st.info("No corrections yet. Review results after analysis to start improving accuracy.")

    st.markdown("---")
    st.markdown("### Confidence Guide")
    st.markdown(
        '<div style="font-size:0.85rem; line-height:1.6;">'
        '<span style="color:#00B050; font-weight:700;">HIGH (80-100%)</span><br>'
        'All boxes clearly visible, brands and SKUs confidently identified. Minimal review needed.<br><br>'
        '<span style="color:#FF8C00; font-weight:700;">MEDIUM (55-79%)</span><br>'
        'Some boxes obscured, blurry, or hard to read. Review recommended — AI may have missed or misidentified some brands.<br><br>'
        '<span style="color:#FF0000; font-weight:700;">LOW (0-54%)</span><br>'
        'Significant visibility issues. Many boxes unreadable or unidentified. Manual review required.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Upload the CHHAT Excel file\n"
        "2. Click **Run Analysis**\n"
        "3. **Review & Correct** the results\n"
        "4. System learns from your corrections\n"
        "5. Download improved results"
    )

# ── Main Content ────────────────────────────────────────────────────────────
st.image("chhat-logo.png", width=180)
st.markdown('<div class="main-header">CHHAT AI Brand Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by AI — Cigarette brand detection from outlet survey photos</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CHHAT Excel file (.xlsx)", type=["xlsx", "xlsm"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Validate file
    from validation import validate_excel_file, validate_api_keys, estimate_processing_time

    file_check = validate_excel_file(tmp_path)
    if not file_check["valid"]:
        st.error(f"Invalid file: {file_check['error']}")
        st.stop()

    # Validate API key
    key_check = validate_api_keys(model)
    if not key_check["valid"]:
        st.error(f"API key issue: {key_check['error']}")
        st.stop()

    try:
        rows = read_raw_data(tmp_path, start_row=start_row, photo_cols_str=photo_cols)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Preview
    st.markdown("### Data Preview")
    preview = [{"Serial": r["serial"], "Images": len(r["urls"]), "First URL": r["urls"][0][:60] + "..." if r["urls"] else "None"} for r in rows]
    st.dataframe(pd.DataFrame(preview), use_container_width=True, height=min(len(rows) * 40 + 40, 300))

    total_images = sum(len(r["urls"]) for r in rows)
    time_est = estimate_processing_time(total_images, model=model, enhancements_enabled=enable_ocr or enable_sku_refinement)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Outlets", len(rows))
    c2.metric("Total Images", total_images)
    c3.metric("Est. Time", time_est["display"])
    st.caption(f"Using {model} at {time_est['rpm']} RPM with {time_est['workers']} parallel workers. ~{time_est['per_image_seconds']}s per image.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        api_keys = {
            "claude": os.getenv("ANTHROPIC_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
            "fireworks": os.getenv("FIREWORKS_API_KEY", ""),
        }

        # Load past corrections for few-shot learning
        recent_corrections = find_relevant_corrections(limit=5)
        correction_context = format_corrections_for_prompt(recent_corrections)
        if recent_corrections:
            st.info(f"Using {len(recent_corrections)} past corrections to improve accuracy.")

        st.markdown("### Live Analysis")
        progress_bar = st.progress(0, text="Starting batch processing...")
        status_text = st.empty()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rate_limiter import RateLimiter
        from process import _process_outlet

        limiter = RateLimiter(model)
        safe_workers = limiter.get_safe_workers()
        status_text.markdown(f"**Processing {len(rows)} outlets in parallel** ({safe_workers} workers, {limiter.rpm} RPM)")

        # Process ALL outlets in parallel using ThreadPoolExecutor
        all_results = []
        analysis_details = []
        outlet_results = [None] * len(rows)
        completed_count = 0

        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_idx = {}
            for idx, row in enumerate(rows):
                future = executor.submit(
                    _process_outlet, row, model, api_keys, correction_context, limiter,
                    enable_enhancement=enable_enhancement,
                    enable_ocr=enable_ocr,
                    enable_sku_refinement=enable_sku_refinement,
                    cross_validation_models=cross_val_models if enable_cross_val else None,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed_count += 1
                try:
                    outlet_results[idx] = future.result()
                except Exception as e:
                    outlet_results[idx] = {
                        "serial": rows[idx]["serial"], "brands": [], "skus": [],
                        "thumbnails": [], "unidentified_packs": 0,
                        "confidence": "low", "confidence_score": 0, "error": str(e),
                    }
                progress_bar.progress(completed_count / len(rows),
                    text=f"Processed {completed_count}/{len(rows)} outlets...")

        progress_bar.progress(1.0, text="Analysis complete!")
        status_text.empty()

        # Display results
        container = st.container()
        for i, row in enumerate(rows):
            result = outlet_results[i]
            if result is None:
                continue
            serial = result["serial"]
            brands_sorted = result.get("brands", [])
            skus_sorted = result.get("skus", [])
            thumbnails = result.get("thumbnails", [])
            total_unidentified = result.get("unidentified_packs", 0)
            error = result.get("error")
            per_image_results = result.get("per_image", [])

            # Compute confidence
            bpi = [r.get("brands", []) for r in per_image_results] if per_image_results else None
            worst_confidence = result.get("confidence", "medium")

            with container:
                with st.container():
                    urls = row["urls"]
                    if thumbnails:
                        img_cols = st.columns(min(len(thumbnails), 3))
                        for url_idx, thumb in enumerate(thumbnails[:3]):
                            with img_cols[url_idx]:
                                st.image(thumb, caption=f"Serial {serial} - Image {url_idx+1}", use_container_width=True)

                    # Use results from parallel processing
                    conf_result = compute_confidence(
                        ai_confidence=worst_confidence,
                        brands_found=brands_sorted,
                        skus_found=skus_sorted,
                        unidentified_packs=total_unidentified,
                        num_images=len(urls),
                        brands_per_image=bpi,
                    )
                    conf = conf_result["level"]
                    conf_score = conf_result["score"]
                    conf_class = f"status-{conf}"
                    card_class = "success" if brands_sorted else ("error" if error else "success")

                    # Build brand+SKU grouped tags
                    brand_sku_map = {}
                    for b in brands_sorted:
                        brand_sku_map[b] = []
                    for s in skus_sorted:
                        # Try to find which brand this SKU belongs to
                        matched = False
                        for b in brands_sorted:
                            if s in BRANDS_AND_SKUS.get(b, []):
                                brand_sku_map[b].append(s)
                                matched = True
                                break
                        if not matched:
                            # Orphan SKU — just list it
                            brand_sku_map.setdefault("_orphan", []).append(s)

                    grouped_tags = ""
                    if brands_sorted:
                        for b in brands_sorted:
                            grouped_tags += f'<span class="brand-sku-group"><span class="brand-tag">{b}</span>'
                            for s in brand_sku_map.get(b, []):
                                grouped_tags += f'<span class="sku-tag">{s}</span>'
                            grouped_tags += '</span> '
                        for s in brand_sku_map.get("_orphan", []):
                            grouped_tags += f'<span class="sku-tag">{s}</span> '
                    else:
                        grouped_tags = '<span style="color:#999">No cigarette brands detected</span>'

                    unid_html = f'<div><div class="metric-label">Unidentified</div><div style="font-size:1.2rem; font-weight:600; color:#FF8C00;">{total_unidentified}</div></div>' if total_unidentified else ''

                    # Confidence bar
                    conf_bar_html = (
                        f'<div class="conf-bar-bg">'
                        f'<div class="conf-bar-fill {conf}" style="width:{conf_score}%"></div>'
                        f'</div>'
                    )

                    st.markdown(
                        f'<div class="result-card {card_class}">'
                        f'<div style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">'
                        f'  <div><div class="metric-label">Serial</div><div style="font-size:1.2rem; font-weight:600;">{serial}</div></div>'
                        f'  <div><div class="metric-label">Brand Count</div><div class="metric-value">{len(brands_sorted)}</div></div>'
                        f'  <div style="min-width:140px;"><div class="metric-label">Confidence</div>'
                        f'    <span class="status-badge {conf_class}">{conf} ({conf_score}%)</span>'
                        f'    {conf_bar_html}'
                        f'  </div>'
                        f'  {unid_html}'
                        f'</div>'
                        f'<div style="margin-top:0.8rem;"><div class="metric-label">Brands & SKUs</div>{grouped_tags}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            result_entry = {
                "serial": serial,
                "brands": brands_sorted,
                "skus": skus_sorted,
                "thumbnails": thumbnails,
                "unidentified_packs": total_unidentified,
                "confidence": conf_result["level"],
                "confidence_score": conf_result["score"],
                "error": error if not brands_sorted and error else None,
            }
            all_results.append(result_entry)
            analysis_details.append({
                "serial": serial,
                "ai_brands": brands_sorted,
                "ai_skus": skus_sorted,
                "per_image": per_image_results,
                "urls": urls,
            })

            if i < len(rows) - 1:
                time.sleep(delay)

        progress_bar.progress(1.0, text="Analysis complete!")
        status_text.empty()

        # Store in session state for the correction UI
        st.session_state["results"] = all_results
        st.session_state["analysis_details"] = analysis_details
        st.session_state["analysis_done"] = True

        # Summary
        st.markdown("### Summary")
        all_brands_global = set()
        for r in all_results:
            all_brands_global.update(r["brands"])

        s1, s2, s3 = st.columns(3)
        s1.metric("Outlets Processed", len(all_results))
        s2.metric("Outlets with Brands", len([r for r in all_results if r["brands"]]))
        s3.metric("Unique Brands Found", len(all_brands_global))

        if all_brands_global:
            brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in sorted(all_brands_global))
            st.markdown(f"**All brands detected:** {brand_html}", unsafe_allow_html=True)

    # ── Review & Correct Section ─────────────────────────────────────────

    if st.session_state.get("analysis_done"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Review & Correct Results")

        all_results = st.session_state["results"]
        analysis_details = st.session_state["analysis_details"]
        all_brand_names = sorted(BRANDS_AND_SKUS.keys())

        # Counter at top of review section
        total_outlets = len(all_results)
        st.markdown(
            f'<div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:10px; '
            f'padding:0.8rem 1.2rem; margin-bottom:1.2rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">'
            f'<span style="font-size:1rem; font-weight:600; color:#3730a3;">'
            f'Reviewing {total_outlets} outlet{"s" if total_outlets != 1 else ""}</span>'
            f'<span style="font-size:0.85rem; color:#6c757d;">Correct any AI errors below. Your corrections improve future analyses.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        corrections_to_save = []

        with st.form("correction_form"):
            for idx, (result, detail) in enumerate(zip(all_results, analysis_details)):
                serial = result["serial"]
                ai_brands = detail["ai_brands"]
                ai_skus = detail["ai_skus"]
                conf_level = result.get("confidence", "medium")
                conf_score = result.get("confidence_score", 0)

                # Determine visual indicator
                if result.get("error"):
                    indicator_html = '<span class="review-indicator indicator-red">&#10008; Error</span>'
                elif conf_level == "low":
                    indicator_html = '<span class="review-indicator indicator-orange">&#9888; Low Confidence</span>'
                elif conf_level == "medium":
                    indicator_html = '<span class="review-indicator indicator-orange">&#9888; Medium Confidence</span>'
                else:
                    indicator_html = '<span class="review-indicator indicator-green">&#10004; High Confidence</span>'

                # Brand tags for summary
                brand_tags_html = ""
                if ai_brands:
                    brand_tags_html = " ".join(f'<span class="brand-tag">{b}</span>' for b in ai_brands)
                else:
                    brand_tags_html = '<span style="color:#999; font-size:0.85rem;">No brands detected</span>'

                # Confidence bar for summary
                conf_bar_class = conf_level if conf_level in ("high", "medium", "low") else "medium"
                summary_conf_bar = (
                    f'<div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.3rem;">'
                    f'<span class="status-badge status-{conf_bar_class}">{conf_level} ({conf_score}%)</span>'
                    f'<div class="conf-bar-bg" style="flex:1; max-width:120px;">'
                    f'<div class="conf-bar-fill {conf_bar_class}" style="width:{conf_score}%"></div>'
                    f'</div></div>'
                )

                with st.expander(f"Serial {serial} — {len(ai_brands)} brand{'s' if len(ai_brands) != 1 else ''} detected", expanded=False):
                    # Summary card before controls
                    st.markdown(
                        f'<div class="review-summary-card">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">'
                        f'  <div><span class="review-serial">#{serial}</span></div>'
                        f'  {indicator_html}'
                        f'</div>'
                        f'<div style="margin-top:0.6rem;">{brand_tags_html}</div>'
                        f'{summary_conf_bar}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    col_brands, col_skus = st.columns(2)

                    with col_brands:
                        st.markdown("**Correct Brands:**")
                        st.caption("Add missing brands or remove incorrect ones")
                        corrected_brands = st.multiselect(
                            "Brands",
                            options=all_brand_names,
                            default=ai_brands,
                            key=f"brands_{serial}_{idx}",
                            label_visibility="collapsed",
                        )

                    with col_skus:
                        st.markdown("**Correct SKUs:**")
                        st.caption("Select specific product variants")

                        # Build SKU options grouped by brand
                        available_skus = []
                        for b in corrected_brands:
                            brand_skus = BRANDS_AND_SKUS.get(b, [])
                            available_skus.extend(brand_skus)

                        # Keep AI SKUs that are still valid
                        valid_ai_skus = [s for s in ai_skus if s in available_skus]

                        corrected_skus = st.multiselect(
                            "SKUs",
                            options=sorted(available_skus),
                            default=valid_ai_skus,
                            key=f"skus_{serial}_{idx}",
                            label_visibility="collapsed",
                        )

                        # Show SKUs grouped by brand for clarity
                        if corrected_brands and available_skus:
                            sku_guide_parts = []
                            for b in corrected_brands:
                                b_skus = BRANDS_AND_SKUS.get(b, [])
                                if b_skus:
                                    sku_list = ", ".join(b_skus[:5])
                                    extra = f" +{len(b_skus)-5} more" if len(b_skus) > 5 else ""
                                    sku_guide_parts.append(f"**{b}:** {sku_list}{extra}")
                            if sku_guide_parts:
                                with st.popover("View SKUs by brand"):
                                    for part in sku_guide_parts:
                                        st.markdown(part)

                    notes = st.text_input(
                        "Notes (optional — helps the AI learn)",
                        placeholder="e.g., 'Small blue ESSE pack hidden behind MEVIUS stack on right side'",
                        key=f"notes_{serial}_{idx}",
                    )

                    # Show diff in colored boxes
                    added_brands = set(corrected_brands) - set(ai_brands)
                    removed_brands = set(ai_brands) - set(corrected_brands)
                    added_skus = set(corrected_skus) - set(ai_skus)
                    removed_skus = set(ai_skus) - set(corrected_skus)

                    has_changes = added_brands or removed_brands or added_skus or removed_skus

                    if has_changes:
                        if added_brands or added_skus:
                            added_parts = []
                            if added_brands:
                                added_parts.append(f"Brands: {', '.join(sorted(added_brands))}")
                            if added_skus:
                                added_parts.append(f"SKUs: {', '.join(sorted(added_skus))}")
                            st.markdown(
                                f'<div class="diff-box-added">&#43; Added by reviewer: {" | ".join(added_parts)}</div>',
                                unsafe_allow_html=True,
                            )
                        if removed_brands or removed_skus:
                            removed_parts = []
                            if removed_brands:
                                removed_parts.append(f"Brands: {', '.join(sorted(removed_brands))}")
                            if removed_skus:
                                removed_parts.append(f"SKUs: {', '.join(sorted(removed_skus))}")
                            st.markdown(
                                f'<div class="diff-box-removed">&#8722; Removed by reviewer: {" | ".join(removed_parts)}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<div class="diff-box-confirmed">Confirmed as correct — no changes</div>',
                            unsafe_allow_html=True,
                        )

                    # Queue correction if something changed
                    if set(corrected_brands) != set(ai_brands) or set(corrected_skus) != set(ai_skus):
                        for url in detail.get("urls", [""]):
                            corrections_to_save.append({
                                "serial": str(serial),
                                "image_url": url,
                                "model_used": model,
                                "ai_result": {"brands": ai_brands, "skus": ai_skus},
                                "corrected_result": {"brands": corrected_brands, "skus": corrected_skus},
                                "notes": notes,
                            })

            submitted = st.form_submit_button("Save Corrections & Improve System", type="primary", use_container_width=True)

            if submitted:
                if corrections_to_save:
                    for c in corrections_to_save:
                        save_correction(c)
                    st.success(f"Saved {len(corrections_to_save)} corrections. The system will use these to improve future analyses.")
                    # Update results with corrections
                    for detail in analysis_details:
                        serial = detail["serial"]
                        for idx2, r in enumerate(all_results):
                            if r["serial"] == serial:
                                key_b = f"brands_{serial}_{idx2}"
                                key_s = f"skus_{serial}_{idx2}"
                                if key_b in st.session_state:
                                    all_results[idx2]["brands"] = sorted(st.session_state[key_b])
                                if key_s in st.session_state:
                                    all_results[idx2]["skus"] = sorted(st.session_state[key_s])
                    st.session_state["results"] = all_results
                else:
                    st.info("No changes detected. All results confirmed as correct.")

        # ── Download ─────────────────────────────────────────────────────

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Download Results")

        results_data = st.session_state["results"]
        stem = Path(uploaded_file.name).stem

        # Build both exports upfront
        detailed_buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out_tmp:
            build_output(results_data, out_tmp.name)
            with open(out_tmp.name, "rb") as f:
                detailed_buffer.write(f.read())
            os.unlink(out_tmp.name)

        from process import build_client_format
        client_buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out_tmp:
            build_client_format(results_data, out_tmp.name)
            with open(out_tmp.name, "rb") as f:
                client_buffer.write(f.read())
            os.unlink(out_tmp.name)

        dl1, dl2 = st.columns(2)

        with dl1:
            st.markdown(
                '<div class="download-card">'
                '<h4>Detailed Report</h4>'
                '<p>Full analysis with images, confidence scores, SKUs, and unidentified pack counts.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download Detailed Report",
                data=detailed_buffer.getvalue(),
                file_name=f"{stem}_detailed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                key="dl_detailed",
            )

        with dl2:
            st.markdown(
                '<div class="download-card">'
                '<h4>Client Format (Q12A + Q12B)</h4>'
                '<p>Clean export with Q12A brand matrix and Q12B SKU codes only — ready for delivery.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download Client Format",
                data=client_buffer.getvalue(),
                file_name=f"{stem}_Q12AB.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                key="dl_client",
            )

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-footer">CHHAT AI Brand Analyzer v1.0 — Powered by Gemini Vision AI</div>',
        unsafe_allow_html=True,
    )

    try:
        os.unlink(tmp_path)
    except (OSError, UnboundLocalError):
        pass
