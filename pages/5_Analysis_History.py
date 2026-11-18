"""
MediVision AI - Page 5: Analysis History & Audit Trail
Search, filter, inspect past evaluations, re-download research PDFs, and manage SQLite records.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import json

from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner
from database.db import db
from database.history import history_manager

st.set_page_config(
    page_title="Analysis History | MediVision AI",
    page_icon="📋",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.title("📋 Analysis History & Audit Log")
st.markdown("Search, filter, and inspect past clinical image analyses stored in the SQLite database.")
render_disclaimer_banner()

# Step 1: Search & Filter Toolbar
col_search, col_cat, col_limit = st.columns([2, 1, 1])

with col_search:
    search_query = st.text_input("🔍 Search by Analysis ID or Filename:", placeholder="e.g. MVA-2026 or sample_01")

with col_cat:
    cat_filter = st.selectbox("Filter Category:", ["ALL", "NORMAL", "PNEUMONIA"], index=0)
    category_param = "" if cat_filter == "ALL" else cat_filter

with col_limit:
    record_limit = st.selectbox("Max Records:", [25, 50, 100, 200], index=1)

# Step 2: Fetch Records from SQLite
records = db.list_analyses(limit=record_limit, search_query=search_query, category=category_param)

if not records:
    st.info("No matching analysis records found in the database.")
else:
    df_history = pd.DataFrame(records)
    st.markdown(f"**Found {len(df_history)} record(s)**")

    # Table View
    display_cols = [
        "analysis_id", "image_filename", "predicted_category",
        "confidence_score", "inference_time_ms", "analysis_timestamp"
    ]
    st.dataframe(df_history[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Step 3: Record Inspection & Actions
    st.markdown("### 🔍 Inspect & Manage Individual Record")
    record_ids = [r["analysis_id"] for r in records]
    selected_id = st.selectbox("Select Analysis ID to inspect:", record_ids)

    selected_record = db.get_analysis_by_id(selected_id)
    if selected_record:
        col_meta, col_img = st.columns([1.5, 1])

        with col_meta:
            cat = selected_record["predicted_category"]
            badge_class = "badge-pneumonia" if cat == "PNEUMONIA" else "badge-normal"
            conf = selected_record["confidence_score"]

            st.markdown(
                f"""
                <div class="med-card">
                    <div style="font-weight: 700; font-size: 16px; color: {settings.COLOR_PRIMARY};">
                        ID: <code>{selected_record['analysis_id']}</code>
                    </div>
                    <div style="margin: 10px 0;">
                        <span class="{badge_class}">{cat}</span>
                        <span style="margin-left: 10px; font-weight: 600; color: #334155;">Confidence: {conf*100:.2f}%</span>
                    </div>
                    <div style="font-size: 13px; color: #64748B; line-height: 1.6;">
                        <b>Image Filename:</b> {selected_record['image_filename']}<br>
                        <b>SHA-256 Hash:</b> <code>{selected_record['image_hash'][:16]}...</code><br>
                        <b>Model:</b> {selected_record['model_name']} (v{selected_record['model_version']})<br>
                        <b>Inference Latency:</b> {selected_record['inference_time_ms']:.1f} ms<br>
                        <b>Timestamp:</b> {selected_record['analysis_timestamp']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Actions row
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                pdf_path = selected_record.get("pdf_report_path")
                if pdf_path and Path(pdf_path).exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Research PDF",
                            data=f.read(),
                            file_name=f"MediVision_{selected_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.caption("PDF Report file not on disk.")

            with act_col2:
                if st.button("🗑️ Delete Analysis Record", type="secondary", use_container_width=True):
                    db.delete_analysis(selected_id, delete_files=True)
                    st.success(f"Analysis '{selected_id}' deleted.")
                    st.rerun()

        with col_img:
            # Display Grad-CAM or Raw Image preview
            gradcam_path = selected_record.get("gradcam_image_path")
            raw_path = selected_record.get("image_path")
            if gradcam_path and Path(gradcam_path).exists():
                st.image(gradcam_path, caption="Grad-CAM Penultimate Attention Heatmap", use_container_width=True)
            elif raw_path and Path(raw_path).exists():
                st.image(raw_path, caption="Original Input Radiograph", use_container_width=True)

        # AI Explanation Accordion
        with st.expander("✨ View Logged AI Explanation", expanded=False):
            ai_json = selected_record.get("ai_explanation_json")
            if ai_json:
                try:
                    parsed = json.loads(ai_json)
                    st.markdown(f"**Executive Summary:** {parsed.get('executive_summary', '')}")
                    st.markdown(f"**Technical Findings:** {parsed.get('technical_findings', '')}")
                    st.markdown(f"**Grad-CAM:** {parsed.get('gradcam_interpretation', '')}")
                except Exception:
                    st.write(ai_json)
            else:
                st.info("No AI explanation recorded for this run.")
