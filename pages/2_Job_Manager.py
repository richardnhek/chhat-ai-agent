"""
Job Manager — view, manage, and download results from all analysis jobs.
"""

import os
import time
import tempfile
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from jobs import get_all_jobs, get_job, get_job_progress, create_job, run_job, delete_job
from image_analyzer import get_available_models
from process import read_raw_data
from corrections import get_correction_stats

load_dotenv()

st.set_page_config(page_title="Job Manager", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .job-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid #4472C4; }
    .job-card.running { border-left-color: #FF8C00; }
    .job-card.completed { border-left-color: #28a745; }
    .job-card.failed { border-left-color: #dc3545; }
    .job-card.queued { border-left-color: #6c757d; }
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("## Job Manager")
st.markdown("Submit new analysis jobs, track progress, and download results.")

# ── New Job Section ──────────────────────────────────────────────────────

st.markdown("### Submit New Job")

col_file, col_settings = st.columns([2, 1])

with col_file:
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xlsm"], key="job_upload")

with col_settings:
    model = st.selectbox("Model", get_available_models(), key="job_model")
    photo_cols = st.text_input("Photo columns", value="B,C,D", key="job_cols")
    start_row = st.number_input("Data starts at row", value=3, min_value=2, key="job_start")

if uploaded and st.button("Submit Job", type="primary", use_container_width=True):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=".") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        rows = read_raw_data(tmp_path, start_row=start_row, photo_cols_str=photo_cols)
        total_images = sum(len(r["urls"]) for r in rows)

        job_id = create_job(
            file_path=tmp_path,
            file_name=uploaded.name,
            model=model,
            start_row=start_row,
            photo_cols=photo_cols,
        )
        run_job(job_id)
        st.success(f"Job **{job_id}** submitted! {len(rows)} outlets, {total_images} images. Processing with {model}...")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# ── Jobs List ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### All Jobs")

jobs = get_all_jobs()
has_running = any(j["status"] == "running" for j in jobs) if jobs else False

if not jobs:
    st.info("No jobs yet. Submit one above.")
else:
    if has_running:
        if st.button("Refresh Progress", use_container_width=True):
            st.rerun()

    for job in reversed(jobs):  # Most recent first
        job_id = job["id"]
        status = job["status"]
        status_emoji = {"queued": "🕐", "running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")

        with st.expander(
            f"{status_emoji} **{job['file_name']}** — {status.upper()} — {job['model']} — {job_id}",
            expanded=(status == "running"),
        ):
            # Info columns
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Outlets", job.get("total_outlets", "?"))
            c2.metric("Images", job.get("total_images", "?"))
            c3.metric("Processed", job.get("processed_outlets", 0))
            c4.metric("Errors", job.get("errors", 0))

            # Timestamps
            created = job.get("created_at", "")[:19]
            started = job.get("started_at", "")[:19] if job.get("started_at") else "—"
            completed_at = job.get("completed_at", "")[:19] if job.get("completed_at") else "—"
            st.caption(f"Created: {created} | Started: {started} | Completed: {completed_at}")

            if status == "running":
                # Show progress
                total = job.get("total_outlets", 1) or 1
                processed = job.get("processed_outlets", 0)
                progress = get_job_progress(job_id)
                pct = processed / total
                st.progress(pct, text=f"Processing {processed}/{total} outlets...")

            elif status == "completed":
                # Show brands found
                brands = job.get("brands_found", [])
                if brands:
                    brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in brands)
                    st.markdown(f"**Brands found ({len(brands)}):** {brand_html}", unsafe_allow_html=True)

                # Download buttons
                d1, d2 = st.columns(2)
                detailed_path = job.get("results_file")
                client_path = job.get("client_format_file")

                if detailed_path and Path(detailed_path).exists():
                    with open(detailed_path, "rb") as f:
                        d1.download_button(
                            "Download Detailed Report",
                            data=f.read(),
                            file_name=f"{Path(job['file_name']).stem}_detailed.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"dl_detailed_{job_id}",
                        )

                if client_path and Path(client_path).exists():
                    with open(client_path, "rb") as f:
                        d2.download_button(
                            "Download Client Format (Q12A+Q12B)",
                            data=f.read(),
                            file_name=f"{Path(job['file_name']).stem}_Q12AB.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"dl_client_{job_id}",
                        )

            elif status == "failed":
                st.error(f"Error: {job.get('error_message', 'Unknown error')}")

            # Delete button
            if st.button(f"Delete Job", key=f"del_{job_id}", type="secondary"):
                delete_job(job_id)
                st.rerun()

# Auto-refresh if jobs are running (every 5 seconds)
if has_running:
    st.info("Jobs are running. Page will auto-refresh every 5 seconds.")
    time.sleep(5)
    st.rerun()
