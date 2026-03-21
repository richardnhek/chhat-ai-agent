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

st.set_page_config(
    page_title="CHHAT Brand Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .correction-added { color: #28a745; font-weight: 600; }
    .correction-removed { color: #dc3545; font-weight: 600; text-decoration: line-through; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
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
    # Show correction stats
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

# Main
st.markdown('<div class="main-header">CHHAT Cigarette Brand Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload survey Excel -> AI identifies brands -> Review & correct -> System improves over time.</div>', unsafe_allow_html=True)

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

    st.markdown("---")

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
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()

        all_results = []
        analysis_details = []  # Store per-image details for correction UI
        container = st.container()

        for i, row in enumerate(rows):
            serial = row["serial"]
            urls = row["urls"]
            progress = (i + 1) / len(rows)
            progress_bar.progress(progress, text=f"Processing outlet {i+1} of {len(rows)}...")
            status_text.markdown(f"**Serial {serial}** — {len(urls)} image(s)")

            all_brands = set()
            all_skus = set()
            thumbnails = []
            total_unidentified = 0
            worst_confidence = "high"
            error = None
            confidence_rank = {"low": 0, "medium": 1, "high": 2}
            per_image_results = []

            with container:
                with st.container():
                    if urls:
                        img_cols = st.columns(min(len(urls), 3))

                    for url_idx, url in enumerate(urls):
                        try:
                            image_data, media_type = fetch_image(url)
                            thumbnails.append(create_thumbnail(image_data))

                            if url_idx < 3:
                                with img_cols[url_idx]:
                                    st.image(image_data, caption=f"Serial {serial} - Image {url_idx+1}", use_container_width=True)

                            with st.spinner(f"Analyzing image {url_idx+1}..."):
                                analysis = analyze_image_enhanced(
                                    image_data, media_type,
                                    model=model, api_keys=api_keys,
                                    correction_context=correction_context,
                                    enable_enhancement=enable_enhancement,
                                    enable_ocr=enable_ocr,
                                    enable_sku_refinement=enable_sku_refinement,
                                    enable_cross_validation=enable_cross_val,
                                    cross_validation_models=cross_val_models if enable_cross_val else None,
                                )

                            if "error" not in analysis:
                                img_brands = []
                                img_skus = []
                                for entry in analysis.get("brands_found", []):
                                    brand = entry.get("brand", "")
                                    if brand in BRANDS_AND_SKUS:
                                        all_brands.add(brand)
                                        img_brands.append(brand)
                                    for sku in entry.get("skus", []):
                                        if sku:
                                            all_skus.add(sku)
                                            img_skus.append(sku)
                                total_unidentified += analysis.get("unidentified_packs", 0)
                                img_conf = analysis.get("confidence", "medium")
                                if confidence_rank.get(img_conf, 1) < confidence_rank.get(worst_confidence, 2):
                                    worst_confidence = img_conf
                                per_image_results.append({
                                    "url": url,
                                    "brands": img_brands,
                                    "skus": img_skus,
                                    "confidence": img_conf,
                                    "unidentified": analysis.get("unidentified_packs", 0),
                                })
                            else:
                                error = analysis.get("error")

                        except Exception as e:
                            error = str(e)

                        if url_idx < len(urls) - 1:
                            time.sleep(delay)

                    brands_sorted = sorted(all_brands)
                    skus_sorted = sorted(all_skus)
                    brand_tags = "".join(f'<span class="brand-tag">{b}</span>' for b in brands_sorted) if brands_sorted else '<span style="color:#999">No cigarette brands detected</span>'
                    sku_tags = "".join(f'<span class="sku-tag">{s}</span>' for s in skus_sorted) if skus_sorted else ''

                    # Compute multi-factor confidence
                    # Collect brands per image for consistency check
                    bpi = [r.get("brands", []) for r in per_image_results] if per_image_results else None
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

                    unid_html = f'<div><div class="metric-label">Unidentified</div><div style="font-size:1.2rem; font-weight:600; color:#FF8C00;">{total_unidentified}</div></div>' if total_unidentified else ''

                    st.markdown(
                        f'<div class="result-card {card_class}">'
                        f'<div style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">'
                        f'  <div><div class="metric-label">Serial</div><div style="font-size:1.2rem; font-weight:600;">{serial}</div></div>'
                        f'  <div><div class="metric-label">Brand Count</div><div class="metric-value">{len(brands_sorted)}</div></div>'
                        f'  <div><div class="metric-label">Confidence</div><span class="status-badge {conf_class}">{conf} ({conf_score}%)</span></div>'
                        f'  {unid_html}'
                        f'</div>'
                        f'<div style="margin-top:0.8rem;"><div class="metric-label">Brands</div>{brand_tags}</div>'
                        + (f'<div style="margin-top:0.5rem;"><div class="metric-label">SKUs</div>{sku_tags}</div>' if sku_tags else '')
                        + f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

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
        st.markdown("---")
        st.markdown("### Review & Correct Results")
        st.markdown("Correct any errors below. Your corrections will improve future analyses.")

        all_results = st.session_state["results"]
        analysis_details = st.session_state["analysis_details"]
        all_brand_names = sorted(BRANDS_AND_SKUS.keys())

        corrections_to_save = []

        with st.form("correction_form"):
            for idx, (result, detail) in enumerate(zip(all_results, analysis_details)):
                serial = result["serial"]
                ai_brands = detail["ai_brands"]
                ai_skus = detail["ai_skus"]

                with st.expander(f"Serial {serial} — AI found {len(ai_brands)} brands: {', '.join(ai_brands) if ai_brands else 'None'}", expanded=False):
                    col_brands, col_skus = st.columns(2)

                    with col_brands:
                        corrected_brands = st.multiselect(
                            "Brands (add/remove as needed)",
                            options=all_brand_names,
                            default=ai_brands,
                            key=f"brands_{serial}_{idx}",
                        )

                    with col_skus:
                        # Build SKU options from selected brands
                        available_skus = []
                        for b in corrected_brands:
                            available_skus.extend(BRANDS_AND_SKUS.get(b, []))

                        # Keep AI SKUs that are still valid
                        valid_ai_skus = [s for s in ai_skus if s in available_skus]

                        corrected_skus = st.multiselect(
                            "SKUs (select specific variants)",
                            options=sorted(available_skus),
                            default=valid_ai_skus,
                            key=f"skus_{serial}_{idx}",
                        )

                    notes = st.text_input(
                        "Notes (optional — helps the AI learn)",
                        placeholder="e.g., 'AI confused gold ESSE for ORIS' or 'blurry pack in corner was CAMBO'",
                        key=f"notes_{serial}_{idx}",
                    )

                    # Show diff
                    added_brands = set(corrected_brands) - set(ai_brands)
                    removed_brands = set(ai_brands) - set(corrected_brands)
                    added_skus = set(corrected_skus) - set(ai_skus)
                    removed_skus = set(ai_skus) - set(corrected_skus)

                    if added_brands or removed_brands or added_skus or removed_skus:
                        diff_parts = []
                        if added_brands:
                            diff_parts.append(f'<span class="correction-added">+ {", ".join(added_brands)}</span>')
                        if removed_brands:
                            diff_parts.append(f'<span class="correction-removed">- {", ".join(removed_brands)}</span>')
                        if added_skus:
                            diff_parts.append(f'<span class="correction-added">SKU + {", ".join(added_skus)}</span>')
                        if removed_skus:
                            diff_parts.append(f'<span class="correction-removed">SKU - {", ".join(removed_skus)}</span>')
                        st.markdown(f"Changes: {' | '.join(diff_parts)}", unsafe_allow_html=True)

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

        st.markdown("---")
        st.markdown("### Download Results")

        export_format = st.radio(
            "Choose export format:",
            ["Detailed Report (with images, confidence, unidentified packs)", "Client Format (Q12A + Q12B only)"],
            horizontal=True,
        )

        results_data = st.session_state["results"]
        stem = Path(uploaded_file.name).stem

        if export_format.startswith("Detailed"):
            output_buffer = io.BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out_tmp:
                build_output(results_data, out_tmp.name)
                with open(out_tmp.name, "rb") as f:
                    output_buffer.write(f.read())
                os.unlink(out_tmp.name)
            st.download_button(
                label="Download Detailed Report",
                data=output_buffer.getvalue(),
                file_name=f"{stem}_detailed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        else:
            from process import build_client_format
            output_buffer = io.BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out_tmp:
                build_client_format(results_data, out_tmp.name)
                with open(out_tmp.name, "rb") as f:
                    output_buffer.write(f.read())
                os.unlink(out_tmp.name)
            st.download_button(
                label="Download Client Format (Q12A + Q12B)",
                data=output_buffer.getvalue(),
                file_name=f"{stem}_Q12AB.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )

    try:
        os.unlink(tmp_path)
    except (OSError, UnboundLocalError):
        pass
