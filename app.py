"""
CHHAT Cigarette Brand Analyzer — Streamlit Dashboard
=====================================================
Upload the CHHAT Excel file → AI identifies cigarette brands in each outlet photo
→ Download results with embedded images + brand/SKU data.
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
from brands import format_q12a, BRANDS_AND_SKUS, BRAND_KHMER
from process import read_raw_data, create_thumbnail, build_output

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
    st.markdown(
        "**How it works:**\n"
        "1. Upload the CHHAT Excel file\n"
        "2. Click **Run Analysis**\n"
        "3. AI scans each outlet photo\n"
        "4. Download results with brands + images"
    )

# Main
st.markdown('<div class="main-header">CHHAT Cigarette Brand Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload the survey Excel → AI identifies cigarette brands from outlet photos → Download results.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CHHAT Excel file (.xlsx)", type=["xlsx", "xlsm"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        rows = read_raw_data(tmp_path, start_row=start_row, photo_cols_str=photo_cols)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Preview
    st.markdown("### Data Preview")
    preview = [{"Serial": r["serial"], "Images": len(r["urls"]), "First URL": r["urls"][0][:60] + "..." if r["urls"] else "None"} for r in rows]
    st.dataframe(pd.DataFrame(preview), use_container_width=True, height=min(len(rows) * 40 + 40, 300))

    c1, c2 = st.columns(2)
    c1.metric("Total Outlets", len(rows))
    c2.metric("Total Images", sum(len(r["urls"]) for r in rows))

    st.markdown("---")

    if st.button("Run Analysis", type="primary", use_container_width=True):
        api_keys = {
            "claude": os.getenv("ANTHROPIC_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
            "fireworks": os.getenv("FIREWORKS_API_KEY", ""),
        }

        st.markdown("### Live Analysis")
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()

        results = []
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
            error = None

            with container:
                with st.container():
                    # Show images in columns
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
                                analysis = analyze_image(image_data, media_type, model=model, api_keys=api_keys)

                            if "error" not in analysis:
                                for entry in analysis.get("brands_found", []):
                                    brand = entry.get("brand", "")
                                    if brand in BRANDS_AND_SKUS:
                                        all_brands.add(brand)
                                    for sku in entry.get("skus", []):
                                        if sku:
                                            all_skus.add(sku)
                            else:
                                error = analysis.get("error")

                        except Exception as e:
                            error = str(e)

                        if url_idx < len(urls) - 1:
                            time.sleep(delay)

                    # Show result card
                    brands_sorted = sorted(all_brands)
                    skus_sorted = sorted(all_skus)
                    brand_tags = "".join(f'<span class="brand-tag">{b}</span>' for b in brands_sorted) if brands_sorted else '<span style="color:#999">No cigarette brands detected</span>'
                    sku_tags = "".join(f'<span class="sku-tag">{s}</span>' for s in skus_sorted) if skus_sorted else ''
                    conf = analysis.get("confidence", "medium") if not error else "low"
                    conf_class = f"status-{conf}"
                    card_class = "success" if brands_sorted else ("error" if error else "success")

                    st.markdown(
                        f'<div class="result-card {card_class}">'
                        f'<div style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">'
                        f'  <div><div class="metric-label">Serial</div><div style="font-size:1.2rem; font-weight:600;">{serial}</div></div>'
                        f'  <div><div class="metric-label">Brand Count</div><div class="metric-value">{len(brands_sorted)}</div></div>'
                        f'  <div><div class="metric-label">Confidence</div><span class="status-badge {conf_class}">{conf}</span></div>'
                        f'</div>'
                        f'<div style="margin-top:0.8rem;"><div class="metric-label">Brands</div>{brand_tags}</div>'
                        + (f'<div style="margin-top:0.5rem;"><div class="metric-label">SKUs</div>{sku_tags}</div>' if sku_tags else '')
                        + f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

            results.append({
                "serial": serial,
                "brands": brands_sorted,
                "skus": skus_sorted,
                "thumbnails": thumbnails,
                "error": error if not brands_sorted and error else None,
            })

            if i < len(rows) - 1:
                time.sleep(delay)

        progress_bar.progress(1.0, text="Analysis complete!")
        status_text.empty()

        # Summary
        st.markdown("### Summary")
        all_brands_global = set()
        for r in results:
            all_brands_global.update(r["brands"])

        s1, s2, s3 = st.columns(3)
        s1.metric("Outlets Processed", len(results))
        s2.metric("Outlets with Brands", len([r for r in results if r["brands"]]))
        s3.metric("Unique Brands Found", len(all_brands_global))

        if all_brands_global:
            brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in sorted(all_brands_global))
            st.markdown(f"**All brands detected:** {brand_html}", unsafe_allow_html=True)

        # Download
        st.markdown("---")
        output_buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out_tmp:
            build_output(results, out_tmp.name)
            with open(out_tmp.name, "rb") as f:
                output_buffer.write(f.read())
            os.unlink(out_tmp.name)

        st.download_button(
            label="Download Results Excel (with embedded images)",
            data=output_buffer.getvalue(),
            file_name=Path(uploaded_file.name).stem + "_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass
