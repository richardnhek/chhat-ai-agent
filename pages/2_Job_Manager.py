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

from jobs import get_all_jobs, get_job, get_job_progress, create_job, run_job, delete_job, resume_job
from image_analyzer import get_available_models
from process import read_raw_data
from corrections import get_correction_stats
from cost_tracker import get_job_cost

load_dotenv()

from auth import check_auth

st.set_page_config(page_title="Job Manager", page_icon="📋", layout="wide")

if not check_auth():
    st.stop()

st.markdown("""
<style>
    .job-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid #4472C4; }
    .job-card.running { border-left-color: #FF8C00; }
    .job-card.completed { border-left-color: #28a745; }
    .job-card.failed { border-left-color: #dc3545; }
    .job-card.queued { border-left-color: #6c757d; }
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem; }

    /* Job status bar */
    .job-status-bar {
        border-radius: 12px 12px 0 0;
        padding: 0.5rem 1rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: white;
        margin: -1.2rem -1rem 1rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .job-status-bar.completed { background: linear-gradient(135deg, #28a745, #1e7e34); }
    .job-status-bar.running { background: linear-gradient(135deg, #FF8C00, #e07600); }
    .job-status-bar.failed { background: linear-gradient(135deg, #dc3545, #c82333); }
    .job-status-bar.queued { background: linear-gradient(135deg, #6c757d, #545b62); }

    .model-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .job-stats-inline {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .job-stat-item {
        text-align: center;
    }
    .job-stat-item .stat-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.2;
    }
    .job-stat-item .stat-lbl {
        font-size: 0.7rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Upload area styling */
    .upload-zone {
        background: #f0f4ff;
        border: 2px dashed #4472C4;
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-zone-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4472C4;
        margin-bottom: 0.3rem;
    }
    .upload-zone-sub {
        font-size: 0.85rem;
        color: #6c757d;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #6c757d;
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .empty-state-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.3rem;
    }
    .empty-state-sub {
        font-size: 0.9rem;
        color: #adb5bd;
    }

    /* Running progress */
    .progress-container {
        background: #fff3e0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        border: 1px solid #ffe0b2;
    }
    .progress-pct {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FF8C00;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Mobile responsive — tablet */
    @media (max-width: 768px) {
        .brand-tag { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
        .job-card { padding: 0.8rem; }
        .upload-zone { padding: 1rem; }
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
        .brand-tag { font-size: 0.65rem; padding: 0.1rem 0.4rem; }
        .job-card { padding: 0.6rem; margin-bottom: 0.5rem; }
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
st.markdown("## Job Manager")
st.markdown("Submit new analysis jobs, track progress, and download results.")

# ── New Job Section ──────────────────────────────────────────────────────

st.markdown("### Submit New Job")

st.markdown(
    '<div class="upload-zone">'
    '  <div class="upload-zone-title">Upload Excel File</div>'
    '  <div class="upload-zone-sub">Drag and drop your CHHAT survey file (.xlsx or .xlsm) below</div>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xlsm"], key="job_upload", label_visibility="collapsed")

with st.expander("Advanced Settings", expanded=False):
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        model = st.selectbox("Model", get_available_models(), key="job_model")
    with col_s2:
        photo_cols = st.text_input("Photo columns", value="B,C,D", key="job_cols")
    with col_s3:
        start_row = st.number_input("Data starts at row", value=3, min_value=2, key="job_start")

if uploaded:
    # Show estimated time before submitting
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=".") as tmp_preview:
            tmp_preview.write(uploaded.read())
            tmp_preview_path = tmp_preview.name
        uploaded.seek(0)  # Reset for later use

        preview_rows = read_raw_data(tmp_preview_path, start_row=start_row, photo_cols_str=photo_cols)
        preview_images = sum(len(r["urls"]) for r in preview_rows)
        est_minutes = max(1, round(preview_images * 2.5 / 60))

        st.markdown(
            f'<div style="background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem; border-left: 4px solid #4472C4; margin-bottom: 1rem;">'
            f'  <strong>{len(preview_rows)} outlets</strong> with <strong>{preview_images} images</strong> detected. '
            f'  Estimated processing time: <strong>~{est_minutes} min</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        try:
            os.unlink(tmp_preview_path)
        except OSError:
            pass
    except Exception:
        pass

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
    st.markdown(
        '<div class="empty-state">'
        '  <div class="empty-state-icon">📂</div>'
        '  <div class="empty-state-title">No jobs yet</div>'
        '  <div class="empty-state-sub">Upload an Excel file above to start your first analysis.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    if has_running:
        if st.button("Refresh Progress", use_container_width=True):
            st.rerun()

    for job in reversed(jobs):  # Most recent first
        job_id = job["id"]
        status = job["status"]

        # Calculate elapsed time for completed jobs
        elapsed_str = ""
        if status == "completed" and job.get("started_at") and job.get("completed_at"):
            try:
                started_dt = datetime.fromisoformat(job["started_at"])
                completed_dt = datetime.fromisoformat(job["completed_at"])
                elapsed_secs = (completed_dt - started_dt).total_seconds()
                if elapsed_secs >= 60:
                    elapsed_str = f"{elapsed_secs / 60:.1f} min"
                else:
                    elapsed_str = f"{elapsed_secs:.0f}s"
            except (ValueError, TypeError):
                pass

        with st.expander(
            f"**{job['file_name']}** — {status.upper()} — {job['model']}",
            expanded=(status == "running"),
        ):
            # Status bar at top
            elapsed_html = f'<span style="font-size: 0.7rem; opacity: 0.85;">Elapsed: {elapsed_str}</span>' if elapsed_str else ""
            st.markdown(
                f'<div class="job-status-bar {status}">'
                f'  <span>{status.upper()}</span>'
                f'  <span style="display:flex; gap:0.5rem; align-items:center;">'
                f'    <span class="model-badge">{job["model"]}</span>'
                f'    {elapsed_html}'
                f'  </span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Info stats inline
            cost_info = get_job_cost(job_id)
            cost_str = f"${cost_info['total_cost']:.3f}" if cost_info["total_calls"] > 0 else "—"

            st.markdown(
                f'<div class="job-stats-inline">'
                f'  <div class="job-stat-item"><div class="stat-val">{job.get("total_outlets", "?")}</div><div class="stat-lbl">Outlets</div></div>'
                f'  <div class="job-stat-item"><div class="stat-val">{job.get("total_images", "?")}</div><div class="stat-lbl">Images</div></div>'
                f'  <div class="job-stat-item"><div class="stat-val">{job.get("processed_outlets", 0)}</div><div class="stat-lbl">Processed</div></div>'
                f'  <div class="job-stat-item"><div class="stat-val">{job.get("errors", 0)}</div><div class="stat-lbl">Errors</div></div>'
                f'  <div class="job-stat-item"><div class="stat-val">{cost_str}</div><div class="stat-lbl">API Cost</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Timestamps
            created = job.get("created_at", "")[:19]
            started = job.get("started_at", "")[:19] if job.get("started_at") else "—"
            completed_at = job.get("completed_at", "")[:19] if job.get("completed_at") else "—"
            st.caption(f"Created: {created} | Started: {started} | Completed: {completed_at}")

            if status == "running":
                # Prominent progress display
                total = job.get("total_outlets", 1) or 1
                processed = job.get("processed_outlets", 0)
                progress = get_job_progress(job_id)
                pct = processed / total
                pct_display = int(pct * 100)

                st.markdown(
                    f'<div class="progress-container">'
                    f'  <div style="display: flex; align-items: center; gap: 1rem;">'
                    f'    <div class="progress-pct">{pct_display}%</div>'
                    f'    <div style="flex: 1; font-size: 0.9rem; color: #495057;">Processing outlet {processed} of {total}</div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(pct)

            elif status == "completed":
                # Show brands found inline
                brands = job.get("brands_found", [])
                total_outlets = job.get("total_outlets", "?")
                if brands:
                    st.markdown(
                        f'<div style="background: #d4edda; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.5rem 0; font-size: 0.85rem; color: #155724;">'
                        f'  <strong>{len(brands)} brands</strong> found across <strong>{total_outlets} outlets</strong>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in brands)
                    st.markdown(f"{brand_html}", unsafe_allow_html=True)

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
                last_serial = job.get("last_processed_serial")
                processed = job.get("processed_outlets", 0)
                total = job.get("total_outlets", 0)
                if processed > 0 and processed < total:
                    st.info(f"Progress saved: {processed}/{total} outlets processed (last: #{last_serial}).")
                if st.button("Resume Job", key=f"resume_{job_id}", type="primary"):
                    resume_job(job_id)
                    st.success(f"Resuming job **{job_id}** from outlet #{last_serial or 'start'}...")
                    time.sleep(1)
                    st.rerun()

            # Delete button
            if st.button("Delete Job", key=f"del_{job_id}", type="secondary"):
                delete_job(job_id)
                st.rerun()

# Auto-refresh if jobs are running (every 5 seconds)
if has_running:
    st.info("Jobs are running. Page will auto-refresh every 5 seconds.")
    time.sleep(5)
    st.rerun()
