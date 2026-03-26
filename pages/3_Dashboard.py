"""
Dashboard — analytics, accuracy stats, confusion matrix, and correction history.
"""

import streamlit as st
from dotenv import load_dotenv

from stats import get_accuracy_stats, get_confusion_matrix, get_processing_stats, export_corrections_csv
from corrections import get_correction_stats, load_corrections
from cost_tracker import get_total_cost
from image_cache import get_cache_stats
from active_learning import get_annotation_stats
from brands import BRANDS_AND_SKUS

load_dotenv()

from auth import check_auth

st.set_page_config(page_title="CHHAT Analytics Dashboard", page_icon="📊", layout="wide")

if not check_auth():
    st.stop()

st.markdown("""
<style>
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem; }
    .stat-card { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .confusion-row { padding: 0.4rem; border-bottom: 1px solid #eee; }
    .dash-metric-card {
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .dash-metric-card .card-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .dash-metric-card .card-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    .health-banner {
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .health-ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .health-warn { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .accuracy-gauge {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #f8f9fa;
        border: 2px solid #dee2e6;
    }
    .accuracy-gauge .gauge-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .accuracy-gauge .gauge-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    .confusion-cell {
        display: inline-block;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        min-width: 60px;
    }
    .learning-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .learning-card .lc-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .learning-card .lc-stat {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .learning-card .lc-label {
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .learning-bar-track {
        background: rgba(255,255,255,0.2);
        border-radius: 6px;
        height: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .learning-bar-fill {
        background: white;
        border-radius: 6px;
        height: 100%;
        transition: width 0.5s ease;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Mobile responsive — tablet */
    @media (max-width: 768px) {
        .brand-tag { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
        .stat-card { padding: 1rem; }
        .dash-metric-card { padding: 0.8rem; }
        .dash-metric-card .card-value { font-size: 1.5rem; }
        .accuracy-gauge .gauge-value { font-size: 2rem; }
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
        .stat-card { padding: 0.6rem; }
        .dash-metric-card { padding: 0.6rem; }
        .dash-metric-card .card-value { font-size: 1.2rem; }
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
st.markdown("## CHHAT Analytics Dashboard")

# ── Processing Stats ────────────────────────────────────────────────────

st.markdown("### Processing Overview")
proc_stats = get_processing_stats()
cost_stats = get_total_cost()

# System Health indicator
error_rate = proc_stats.get("error_rate", 0)
if error_rate > 10:
    st.markdown(
        '<div class="health-banner health-warn">Warning: Error rate is above 10%. Some jobs may need attention.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="health-banner health-ok">All systems operational</div>',
        unsafe_allow_html=True,
    )

# Colored metric cards
total_jobs = proc_stats.get("total_jobs", 0)
completed_jobs = proc_stats.get("completed_jobs", 0)
total_cost_val = f"${cost_stats['total_cost']:.2f}" if cost_stats["total_calls"] > 0 else "$0.00"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="dash-metric-card" style="background: linear-gradient(135deg, #4472C4, #2b5ea7);">'
        f'  <div class="card-value">{total_jobs}</div>'
        f'  <div class="card-label">Total Jobs</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="dash-metric-card" style="background: linear-gradient(135deg, #28a745, #1e7e34);">'
        f'  <div class="card-value">{completed_jobs}</div>'
        f'  <div class="card-label">Completed</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c3:
    err_bg = "linear-gradient(135deg, #FF8C00, #e07600)" if error_rate <= 10 else "linear-gradient(135deg, #dc3545, #c82333)"
    st.markdown(
        f'<div class="dash-metric-card" style="background: {err_bg};">'
        f'  <div class="card-value">{error_rate}%</div>'
        f'  <div class="card-label">Error Rate</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="dash-metric-card" style="background: linear-gradient(135deg, #6f42c1, #5a32a3);">'
        f'  <div class="card-value">{total_cost_val}</div>'
        f'  <div class="card-label">Total API Cost</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Secondary stats row
sc1, sc2 = st.columns(2)
sc1.metric("Outlets Processed", proc_stats.get("total_outlets_processed", 0))
sc2.metric("Images Processed", proc_stats.get("total_images_processed", 0))

brands_found = proc_stats.get("unique_brands_found", [])
if brands_found:
    brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in brands_found)
    st.markdown(f"**All brands detected across all jobs ({len(brands_found)}):** {brand_html}", unsafe_allow_html=True)

models_used = proc_stats.get("models_used", {})
if models_used:
    st.caption(f"Models used: {', '.join(f'{m} ({c}x)' for m, c in models_used.items())}")

if cost_stats["total_calls"] > 0:
    by_model = cost_stats.get("by_model", {})
    if by_model:
        cost_parts = [f"{m}: ${c:.3f}" for m, c in sorted(by_model.items(), key=lambda x: -x[1])]
        st.caption(f"Cost by model: {' | '.join(cost_parts)} | Total API calls: {cost_stats['total_calls']}")

if proc_stats.get("avg_processing_time_seconds"):
    st.caption(f"Average job processing time: {proc_stats['avg_processing_time_seconds']:.0f}s")

# ── Image Cache Stats ──────────────────────────────────────────────────
cache_stats = get_cache_stats()
if cache_stats["total_images"] > 0:
    cc1, cc2 = st.columns(2)
    cc1.metric("Cached Images", cache_stats["total_images"])
    cc2.metric("Cache Size", f"{cache_stats['cache_size_mb']:.1f} MB")

# ── Accuracy Stats ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Accuracy Tracking")

accuracy = get_accuracy_stats()
correction_stats = get_correction_stats()

if accuracy["total_reviewed"] == 0:
    st.info("No corrections submitted yet. Review results after analysis to start tracking accuracy.")
else:
    # Large accuracy gauge
    acc_rate = accuracy.get("accuracy_rate", 0)
    if acc_rate >= 85:
        gauge_color = "#28a745"
    elif acc_rate >= 70:
        gauge_color = "#FF8C00"
    else:
        gauge_color = "#dc3545"

    ga1, ga2, ga3 = st.columns([1, 1, 1])

    with ga1:
        st.markdown(
            f'<div class="accuracy-gauge" style="border-color: {gauge_color};">'
            f'  <div class="gauge-value" style="color: {gauge_color};">{acc_rate}%</div>'
            f'  <div class="gauge-label">AI Accuracy Rate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with ga2:
        st.metric("Outlets Reviewed", accuracy["total_reviewed"])
        st.metric("Corrections Made", accuracy["total_corrected"])

    with ga3:
        st.markdown(
            f'<div style="background: #e8f4fd; border-radius: 8px; padding: 1rem; border-left: 4px solid #4472C4;">'
            f'  <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 0.3rem;">Tip</div>'
            f'  <div style="font-size: 0.85rem; color: #495057;">'
            f'    Submit more corrections to improve accuracy. '
            f'    Current learning: <strong>{correction_stats["total"]} corrections</strong> applied.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Per-brand accuracy
    brand_acc = accuracy.get("brand_accuracy", {})
    if brand_acc:
        st.markdown("#### Brand Detection Accuracy")

        # Sort by accuracy (worst first to highlight problem areas)
        sorted_brands = sorted(brand_acc.items(), key=lambda x: x[1]["accuracy"])

        import pandas as pd

        # Build HTML table with color-coded accuracy
        table_rows = ""
        for brand, data in sorted_brands:
            bacc = data["accuracy"]
            if bacc >= 85:
                acc_style = "color: #155724; background: #d4edda; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;"
            elif bacc >= 70:
                acc_style = "color: #856404; background: #fff3cd; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;"
            else:
                acc_style = "color: #721c24; background: #f8d7da; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;"
            table_rows += (
                f'<tr>'
                f'  <td style="padding: 0.5rem 0.8rem; font-weight: 500;">{brand}</td>'
                f'  <td style="padding: 0.5rem 0.8rem; text-align: center;">{data["correct"]}</td>'
                f'  <td style="padding: 0.5rem 0.8rem; text-align: center;">{data["total"]}</td>'
                f'  <td style="padding: 0.5rem 0.8rem; text-align: center;"><span style="{acc_style}">{bacc}%</span></td>'
                f'</tr>'
            )

        st.markdown(
            f'<table style="width: 100%; border-collapse: collapse; background: #f8f9fa; border-radius: 8px; overflow: hidden;">'
            f'<thead><tr style="background: #e9ecef;">'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem; text-transform: uppercase; color: #495057;">Brand</th>'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase; color: #495057;">Correct</th>'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase; color: #495057;">Total</th>'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase; color: #495057;">Accuracy</th>'
            f'</tr></thead>'
            f'<tbody>{table_rows}</tbody>'
            f'</table>',
            unsafe_allow_html=True,
        )

    # Per-SKU accuracy
    sku_acc = accuracy.get("sku_accuracy", {})
    if sku_acc:
        with st.expander("SKU Detection Accuracy (detailed)"):
            import pandas as pd
            sorted_skus = sorted(sku_acc.items(), key=lambda x: x[1]["accuracy"])
            df_sku = pd.DataFrame([
                {
                    "SKU": sku,
                    "Correct": data["correct"],
                    "Total": data["total"],
                    "Accuracy": f"{data['accuracy']}%",
                }
                for sku, data in sorted_skus
            ])
            st.dataframe(df_sku, use_container_width=True, hide_index=True)

# ── Confusion Matrix ────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Brand Confusion Matrix")

confusion = get_confusion_matrix()

if confusion["total_confusions"] == 0:
    st.info("No brand confusions recorded yet. These will appear as corrections are submitted.")
else:
    st.markdown(
        '<div style="background: #f8f9fa; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #495057;">'
        'This shows which brands the AI confuses with each other. Top confusions are addressed first by the learning system.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.metric("Total Confusions", confusion["total_confusions"])

    # Color-coded confusion table
    confusions_list = confusion["confusions"]
    if confusions_list:
        max_count = max(item["count"] for item in confusions_list) if confusions_list else 1
        table_rows = ""
        for item in confusions_list:
            count = item["count"]
            # Darker red for more confusions
            intensity = min(count / max_count, 1.0)
            r = int(255 - intensity * 55)
            g = int(255 - intensity * 100)
            b_val = int(255 - intensity * 100)
            bg_color = f"rgb({r},{g},{b_val})"
            font_weight = "700" if intensity > 0.5 else "500"
            table_rows += (
                f'<tr style="background: {bg_color};">'
                f'  <td style="padding: 0.5rem 0.8rem; font-weight: 500;">{item["ai_detected"]}</td>'
                f'  <td style="padding: 0.5rem 0.8rem; font-weight: 500;">{item["should_have_been"]}</td>'
                f'  <td style="padding: 0.5rem 0.8rem; text-align: center; font-weight: {font_weight};">{count}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<table style="width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden;">'
            f'<thead><tr style="background: #343a40; color: white;">'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem; text-transform: uppercase;">AI Detected (wrong)</th>'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem; text-transform: uppercase;">Should Have Been</th>'
            f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase;">Times</th>'
            f'</tr></thead>'
            f'<tbody>{table_rows}</tbody>'
            f'</table>',
            unsafe_allow_html=True,
        )

    # Highlight top confusions
    top = confusion["confusions"][:5]
    if top:
        st.markdown("")
        st.markdown("**Top confusion patterns:**")
        for item in top:
            st.markdown(
                f"- AI says **{item['ai_detected']}** but should be **{item['should_have_been']}** "
                f"({item['count']} time{'s' if item['count'] > 1 else ''})"
            )

# ── Learning Progress ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Learning Progress")

correction_stats = get_correction_stats()
total_corrections = correction_stats["total"]

if total_corrections == 0:
    st.info("No corrections submitted yet. The system learns from your corrections to improve brand detection accuracy over time.")
else:
    lp1, lp2 = st.columns([1, 2])

    with lp1:
        # Estimate improvement: rough heuristic
        estimated_improvement = min(total_corrections * 0.5, 25)  # cap at 25%
        st.markdown(
            f'<div class="learning-card">'
            f'  <div class="lc-title">System Learning</div>'
            f'  <div class="lc-stat">{total_corrections}</div>'
            f'  <div class="lc-label">Total Corrections</div>'
            f'  <div style="margin-top: 1rem;">'
            f'    <div class="lc-label">Estimated Improvement</div>'
            f'    <div class="learning-bar-track">'
            f'      <div class="learning-bar-fill" style="width: {min(estimated_improvement * 4, 100)}%;"></div>'
            f'    </div>'
            f'    <div style="font-size: 0.9rem; margin-top: 0.3rem; font-weight: 600;">~{estimated_improvement:.0f}% improvement</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with lp2:
        st.markdown(
            f'<div style="background: #f8f9fa; border-radius: 12px; padding: 1.2rem; height: 100%;">'
            f'  <div style="font-weight: 600; font-size: 1rem; margin-bottom: 0.8rem; color: #1a1a2e;">Correction Breakdown</div>'
            f'  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">'
            f'    <div style="background: white; border-radius: 8px; padding: 0.8rem; text-align: center; border: 1px solid #dee2e6;">'
            f'      <div style="font-size: 1.5rem; font-weight: 700; color: #4472C4;">{correction_stats.get("brand_swaps", 0)}</div>'
            f'      <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase;">Brand Swaps</div>'
            f'    </div>'
            f'    <div style="background: white; border-radius: 8px; padding: 0.8rem; text-align: center; border: 1px solid #dee2e6;">'
            f'      <div style="font-size: 1.5rem; font-weight: 700; color: #28a745;">{correction_stats.get("brand_added", 0)}</div>'
            f'      <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase;">Brands Added</div>'
            f'    </div>'
            f'    <div style="background: white; border-radius: 8px; padding: 0.8rem; text-align: center; border: 1px solid #dee2e6;">'
            f'      <div style="font-size: 1.5rem; font-weight: 700; color: #dc3545;">{correction_stats.get("brand_removed", 0)}</div>'
            f'      <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase;">Brands Removed</div>'
            f'    </div>'
            f'    <div style="background: white; border-radius: 8px; padding: 0.8rem; text-align: center; border: 1px solid #dee2e6;">'
            f'      <div style="font-size: 1.5rem; font-weight: 700; color: #6f42c1;">{correction_stats.get("sku_corrected", 0)}</div>'
            f'      <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase;">SKU Fixes</div>'
            f'    </div>'
            f'  </div>'
            f'  <div style="margin-top: 1rem; font-size: 0.85rem; color: #6c757d;">'
            f'    Accuracy improves as more corrections are submitted. '
            f'    Currently: {total_corrections} corrections applied to the learning system.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Export Corrections ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Export Data")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if correction_stats["total"] > 0:
        csv_data = export_corrections_csv()
        st.download_button(
            f"Export Corrections ({correction_stats['total']} records)",
            data=csv_data,
            file_name="chhat_corrections_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No corrections to export yet.")

with col_exp2:
    st.info("Correction data can be used as training data for future model fine-tuning (Option 2).")

# ══════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION SECTION
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## Model Evaluation")

eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
    "Per-Brand Accuracy",
    "Accuracy Over Time",
    "Error Gallery",
    "Training Data Coverage",
])

# ── Tab 1: Per-Brand Accuracy Chart ──────────────────────────────────────
with eval_tab1:
    accuracy = get_accuracy_stats()
    brand_acc = accuracy.get("brand_accuracy") or {}

    if not brand_acc:
        st.info("No accuracy data yet. Submit corrections after analysis to start tracking per-brand accuracy.")
    else:
        st.markdown("### Per-Brand Accuracy")
        st.caption("Horizontal bars show accuracy % for each brand. Colors: green (90%+), orange (70-89%), red (<70%).")

        # Sort by accuracy ascending so worst brands are at top
        sorted_brands = sorted(brand_acc.items(), key=lambda x: x[1]["accuracy"])

        bar_rows_html = ""
        for brand, data in sorted_brands:
            bacc = data["accuracy"]
            if bacc >= 90:
                bar_color = "#28a745"
            elif bacc >= 70:
                bar_color = "#FF8C00"
            else:
                bar_color = "#dc3545"

            bar_width = max(bacc, 2)  # minimum visible width
            bar_rows_html += (
                f'<div style="display: flex; align-items: center; margin-bottom: 6px;">'
                f'  <div style="min-width: 130px; font-weight: 500; font-size: 0.85rem; color: #1a1a2e;">{brand}</div>'
                f'  <div style="flex: 1; background: #e9ecef; border-radius: 6px; height: 24px; overflow: hidden; position: relative;">'
                f'    <div style="width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 6px; '
                f'         display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; '
                f'         transition: width 0.5s ease;">'
                f'      <span style="color: white; font-size: 0.75rem; font-weight: 700;">{bacc}%</span>'
                f'    </div>'
                f'  </div>'
                f'  <div style="min-width: 70px; text-align: right; font-size: 0.78rem; color: #6c757d;">'
                f'    {data["correct"]}/{data["total"]}'
                f'  </div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background: #f8f9fa; border-radius: 12px; padding: 1.2rem;">{bar_rows_html}</div>',
            unsafe_allow_html=True,
        )

        # Summary stats
        total_brands = len(brand_acc)
        good_brands = len([b for b, d in brand_acc.items() if d["accuracy"] >= 90])
        warn_brands = len([b for b, d in brand_acc.items() if 70 <= d["accuracy"] < 90])
        bad_brands = len([b for b, d in brand_acc.items() if d["accuracy"] < 70])

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Good (90%+)", good_brands)
        sc2.metric("Needs Work (70-89%)", warn_brands)
        sc3.metric("Critical (<70%)", bad_brands)


# ── Tab 2: Accuracy Over Time ────────────────────────────────────────────
with eval_tab2:
    all_corrections = load_corrections()
    real_corrections = [c for c in all_corrections if c.get("correction_type") != "no_change"]

    if not all_corrections:
        st.info("No corrections submitted yet. Accuracy tracking will appear as corrections accumulate.")
    else:
        st.markdown("### Accuracy Improvement Over Time")
        st.caption("Shows how accuracy evolves as more corrections are submitted. "
                   "Each round represents a batch of 10 corrections.")

        # Group corrections into rounds of 10
        round_size = 10
        rounds = []
        for i in range(0, len(all_corrections), round_size):
            batch = all_corrections[:i + round_size]
            batch_real = [c for c in batch if c.get("correction_type") != "no_change"]
            total_reviewed = len(batch)
            total_corrected = len(batch_real)
            acc = ((total_reviewed - total_corrected) / total_reviewed * 100) if total_reviewed > 0 else 0
            rounds.append({
                "round": len(rounds) + 1,
                "total_corrections": i + min(round_size, len(all_corrections) - i),
                "accuracy": round(acc, 1),
            })

        if len(rounds) >= 2:
            import pandas as pd
            df_rounds = pd.DataFrame(rounds)
            df_rounds = df_rounds.set_index("round")
            st.line_chart(df_rounds["accuracy"], use_container_width=True)
        else:
            st.info("Need at least 2 rounds of corrections (20+) to show a trend line.")

        # Show rounds table
        st.markdown("#### Correction Rounds")
        round_rows_html = ""
        for r in rounds:
            round_rows_html += (
                f'<div style="display: flex; justify-content: space-between; padding: 0.5rem 0.8rem; '
                f'border-bottom: 1px solid #eee;">'
                f'  <span style="font-weight: 500;">Round {r["round"]}</span>'
                f'  <span style="color: #6c757d;">{r["total_corrections"]} corrections</span>'
                f'  <span style="font-weight: 600; color: {"#28a745" if r["accuracy"] >= 85 else "#FF8C00" if r["accuracy"] >= 70 else "#dc3545"};">'
                f'    {r["accuracy"]}% accuracy'
                f'  </span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background: #f8f9fa; border-radius: 8px; overflow: hidden;">{round_rows_html}</div>',
            unsafe_allow_html=True,
        )


# ── Tab 3: Error Gallery ────────────────────────────────────────────────
with eval_tab3:
    all_corrections = load_corrections()
    # Errors: corrections where the model was wrong (not human_annotation and not no_change)
    errors = [
        c for c in all_corrections
        if c.get("correction_type") != "no_change"
        and c.get("model_used") != "human_annotation"
    ]

    if not errors:
        st.info("No model errors recorded yet. These appear when you correct AI results during review.")
    else:
        st.markdown("### Recent Model Errors")
        st.caption("The last 10 images where the AI model was wrong and a human corrected the result.")

        # Show most recent 10 errors
        recent_errors = errors[-10:]
        recent_errors.reverse()  # newest first

        for i, err in enumerate(recent_errors):
            ai_brands = err.get("ai_result", {}).get("brands", [])
            correct_brands = err.get("corrected_result", {}).get("brands", [])
            ai_skus = err.get("ai_result", {}).get("skus", [])
            correct_skus = err.get("corrected_result", {}).get("skus", [])
            serial = err.get("serial", "N/A")
            image_url = err.get("image_url", "")
            model = err.get("model_used", "unknown")
            correction_type = err.get("correction_type", "unknown")
            timestamp = err.get("timestamp", "")

            # Color for correction type
            type_colors = {
                "brand_swap": "#dc3545",
                "brand_added": "#28a745",
                "brand_removed": "#FF8C00",
                "sku_corrected": "#6f42c1",
            }
            type_color = type_colors.get(correction_type, "#6c757d")

            # Build brand diff
            ai_set = set(ai_brands)
            correct_set = set(correct_brands)
            false_pos = ai_set - correct_set
            false_neg = correct_set - ai_set
            correct_matches = ai_set & correct_set

            diff_parts = []
            for b in sorted(false_pos):
                diff_parts.append(f'<span style="background:#f8d7da;color:#721c24;padding:0.15rem 0.5rem;border-radius:12px;font-size:0.78rem;margin:0.1rem;">-{b}</span>')
            for b in sorted(false_neg):
                diff_parts.append(f'<span style="background:#d4edda;color:#155724;padding:0.15rem 0.5rem;border-radius:12px;font-size:0.78rem;margin:0.1rem;">+{b}</span>')
            for b in sorted(correct_matches):
                diff_parts.append(f'<span style="background:#e9ecef;color:#495057;padding:0.15rem 0.5rem;border-radius:12px;font-size:0.78rem;margin:0.1rem;">{b}</span>')

            diff_html = " ".join(diff_parts) if diff_parts else "<em>No brands</em>"

            st.markdown(
                f'<div style="background: #f8f9fa; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; '
                f'border-left: 4px solid {type_color};">'
                f'  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
                f'    <span style="font-weight: 600; color: #1a1a2e;">Serial: {serial}</span>'
                f'    <span style="background: {type_color}; color: white; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem;">'
                f'      {correction_type.replace("_", " ").title()}'
                f'    </span>'
                f'  </div>'
                f'  <div style="font-size: 0.85rem; margin-bottom: 0.3rem;">'
                f'    <strong>AI said:</strong> {", ".join(ai_brands) if ai_brands else "none"} '
                f'    <strong style="margin-left: 1rem;">Correct:</strong> {", ".join(correct_brands) if correct_brands else "none"}'
                f'  </div>'
                f'  <div style="margin-bottom: 0.3rem;">{diff_html}</div>'
                f'  <div style="font-size: 0.75rem; color: #6c757d;">Model: {model} | {timestamp[:16] if timestamp else "N/A"}'
                f'    {f" | Image: {image_url[:50]}..." if image_url else ""}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Link to Annotate page
        st.markdown("")
        st.page_link("pages/5_Annotate.py", label="Open Annotate Page to Re-annotate", icon="🎯")


# ── Tab 4: Training Data Coverage ────────────────────────────────────────
with eval_tab4:
    st.markdown("### Training Data Coverage")
    st.caption("Shows how many annotations exist for each brand. "
               "Brands with more data produce better AI results.")

    ann_stats = get_annotation_stats()
    annotations_per_brand = ann_stats.get("annotations_per_brand", {})

    all_brands = sorted(BRANDS_AND_SKUS.keys())

    # Count reference images from corrections
    all_corrections = load_corrections()
    brand_reference_counts = {}
    for brand in all_brands:
        ref_count = 0
        for c in all_corrections:
            corrected_brands = set(c.get("corrected_result", {}).get("brands", []))
            if brand in corrected_brands and c.get("model_used") == "human_annotation":
                ref_count += 1
        brand_reference_counts[brand] = ref_count

    # Build table
    coverage_rows = ""
    for brand in all_brands:
        annotated_count = annotations_per_brand.get(brand, 0)
        ref_count = brand_reference_counts.get(brand, 0)

        if annotated_count >= 20:
            status = "Good"
            status_style = "background: #d4edda; color: #155724; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.78rem;"
        elif annotated_count >= 5:
            status = "Needs More"
            status_style = "background: #fff3cd; color: #856404; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.78rem;"
        else:
            status = "Critical"
            status_style = "background: #f8d7da; color: #721c24; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.78rem;"

        coverage_rows += (
            f'<tr>'
            f'  <td style="padding: 0.5rem 0.8rem; font-weight: 500;">{brand}</td>'
            f'  <td style="padding: 0.5rem 0.8rem; text-align: center;">{annotated_count}</td>'
            f'  <td style="padding: 0.5rem 0.8rem; text-align: center;">{ref_count}</td>'
            f'  <td style="padding: 0.5rem 0.8rem; text-align: center;"><span style="{status_style}">{status}</span></td>'
            f'</tr>'
        )

    st.markdown(
        f'<table style="width: 100%; border-collapse: collapse; background: #f8f9fa; border-radius: 8px; overflow: hidden;">'
        f'<thead><tr style="background: #343a40; color: white;">'
        f'  <th style="padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem; text-transform: uppercase;">Brand</th>'
        f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase;">Annotated Images</th>'
        f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase;">Reference Images</th>'
        f'  <th style="padding: 0.6rem 0.8rem; text-align: center; font-size: 0.8rem; text-transform: uppercase;">Status</th>'
        f'</tr></thead>'
        f'<tbody>{coverage_rows}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )

    # Summary
    good_count = len([b for b in all_brands if annotations_per_brand.get(b, 0) >= 20])
    needs_more_count = len([b for b in all_brands if 5 <= annotations_per_brand.get(b, 0) < 20])
    critical_count = len([b for b in all_brands if annotations_per_brand.get(b, 0) < 5])

    st.markdown("")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Good Coverage", f"{good_count} brands")
    sm2.metric("Needs More", f"{needs_more_count} brands")
    sm3.metric("Critical", f"{critical_count} brands")

    if critical_count > 0:
        st.markdown("")
        st.page_link("pages/10_Smart_Queue.py", label="Open Smart Queue to annotate high-priority images", icon="🧠")
