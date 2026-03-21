"""
Dashboard — analytics, accuracy stats, confusion matrix, and correction history.
"""

import streamlit as st
from dotenv import load_dotenv

from stats import get_accuracy_stats, get_confusion_matrix, get_processing_stats, export_corrections_csv
from corrections import get_correction_stats

load_dotenv()

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem; }
    .stat-card { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .confusion-row { padding: 0.4rem; border-bottom: 1px solid #eee; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("## Dashboard & Analytics")

# ── Processing Stats ────────────────────────────────────────────────────

st.markdown("### Processing Overview")
proc_stats = get_processing_stats()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Jobs", proc_stats.get("total_jobs", 0))
c2.metric("Completed", proc_stats.get("completed_jobs", 0))
c3.metric("Outlets Processed", proc_stats.get("total_outlets_processed", 0))
c4.metric("Images Processed", proc_stats.get("total_images_processed", 0))
c5.metric("Error Rate", f"{proc_stats.get('error_rate', 0)}%")

brands_found = proc_stats.get("unique_brands_found", [])
if brands_found:
    brand_html = "".join(f'<span class="brand-tag">{b}</span>' for b in brands_found)
    st.markdown(f"**All brands detected across all jobs ({len(brands_found)}):** {brand_html}", unsafe_allow_html=True)

models_used = proc_stats.get("models_used", {})
if models_used:
    st.caption(f"Models used: {', '.join(f'{m} ({c}x)' for m, c in models_used.items())}")

if proc_stats.get("avg_processing_time_seconds"):
    st.caption(f"Average job processing time: {proc_stats['avg_processing_time_seconds']:.0f}s")

# ── Accuracy Stats ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Accuracy Tracking")

accuracy = get_accuracy_stats()

if accuracy["total_reviewed"] == 0:
    st.info("No corrections submitted yet. Review results after analysis to start tracking accuracy.")
else:
    a1, a2, a3 = st.columns(3)
    a1.metric("Outlets Reviewed", accuracy["total_reviewed"])
    a2.metric("Corrections Made", accuracy["total_corrected"])
    a3.metric("AI Accuracy Rate", f"{accuracy['accuracy_rate']}%")

    # Per-brand accuracy
    brand_acc = accuracy.get("brand_accuracy", {})
    if brand_acc:
        st.markdown("#### Brand Detection Accuracy")

        # Sort by accuracy (worst first to highlight problem areas)
        sorted_brands = sorted(brand_acc.items(), key=lambda x: x[1]["accuracy"])

        import pandas as pd
        df = pd.DataFrame([
            {
                "Brand": brand,
                "Correct": data["correct"],
                "Total": data["total"],
                "Accuracy": f"{data['accuracy']}%",
            }
            for brand, data in sorted_brands
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-SKU accuracy
    sku_acc = accuracy.get("sku_accuracy", {})
    if sku_acc:
        with st.expander("SKU Detection Accuracy (detailed)"):
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
st.caption("Shows which brands the AI confuses with each other — helps identify systematic errors.")

confusion = get_confusion_matrix()

if confusion["total_confusions"] == 0:
    st.info("No brand confusions recorded yet. These will appear as corrections are submitted.")
else:
    st.metric("Total Confusions", confusion["total_confusions"])

    import pandas as pd
    df_conf = pd.DataFrame(confusion["confusions"])
    df_conf.columns = ["AI Detected (wrong)", "Should Have Been", "Times"]
    st.dataframe(df_conf, use_container_width=True, hide_index=True)

    # Highlight top confusions
    top = confusion["confusions"][:5]
    if top:
        st.markdown("**Top confusion patterns:**")
        for item in top:
            st.markdown(
                f"- AI says **{item['ai_detected']}** but should be **{item['should_have_been']}** "
                f"({item['count']} time{'s' if item['count'] > 1 else ''})"
            )

# ── Export Corrections ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Export Data")

correction_stats = get_correction_stats()

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
