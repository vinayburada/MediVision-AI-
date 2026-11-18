"""
MediVision AI - Page 3: AI-Assisted Educational Explanations
Generative AI reasoning via Google Gemini, prompt inspection, and clinical AI limitations.
"""
from pathlib import Path
import streamlit as st
import json

from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner
from llm.gemini_client import gemini_client
from database.db import db

st.set_page_config(
    page_title="AI Explanation | MediVision AI",
    page_icon="✨",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.title("✨ Generative AI – Educational Explanation")
st.markdown("LLM-assisted educational summaries translating deep learning activations into accessible scientific findings.")
render_disclaimer_banner()

# Check Gemini status
if gemini_client.is_configured:
    st.success("Google Gemini API is connected and active.", icon="✨")
else:
    st.info(
        "Google Gemini API key not found in environment (`.env`). "
        "The educational engine is running in deterministic offline research mode with verified medical AI templates.",
        icon="ℹ️",
    )

# Step 1: Select Analysis to explain
analyses = db.list_analyses(limit=25)

if not analyses and "current_analysis" not in st.session_state:
    st.warning("No analyses recorded yet. Run an analysis on the **1_Image_Analysis** page first.")
else:
    options = {}
    if "current_analysis" in st.session_state:
        cur = st.session_state["current_analysis"]
        options[f"Current Session: {cur['analysis_id']} ({cur['predicted_category']})"] = cur

    for item in analyses:
        key = f"History: {item['analysis_id']} ({item['predicted_category']} - {item['confidence_score']*100:.1f}%)"
        options[key] = item

    selected_key = st.selectbox("Select Analysis Record to Review:", list(options.keys()))
    record = options[selected_key]

    # Normalize fields whether from session_state or database
    analysis_id = record["analysis_id"]
    predicted_cat = record["predicted_category"]
    confidence = float(record["confidence_score"])
    normal_p = float(record.get("normal_probability", record.get("class_probabilities", {}).get("NORMAL", 0.5)))
    pneumonia_p = float(record.get("pneumonia_probability", record.get("class_probabilities", {}).get("PNEUMONIA", 0.5)))
    model_name = record.get("model_name", "DenseNet121")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Predicted Category", predicted_cat)
    with m2:
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")
    with m3:
        st.metric("Normal Score", f"{normal_p * 100:.2f}%")
    with m4:
        st.metric("Pneumonia Score", f"{pneumonia_p * 100:.2f}%")

    st.write("")

    # Button to re-generate explanation via Gemini live
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        regen = st.button("🔄 Generate Live Gemini Explanation", type="primary")

    ai_data = None
    if regen:
        with st.spinner("Querying Google Gemini API..."):
            ai_data = gemini_client.generate_educational_explanation(
                predicted_category=predicted_cat,
                confidence_score=confidence,
                normal_probability=normal_p,
                pneumonia_probability=pneumonia_p,
                model_name=model_name,
                gradcam_description="Concentrated spatial activations localized to parenchymal fields.",
            )
    else:
        # Load from saved record if present
        if "ai_explanation" in record:
            ai_data = record["ai_explanation"]
        elif record.get("ai_explanation_json"):
            try:
                ai_data = json.loads(record["ai_explanation_json"])
            except Exception:
                ai_data = None

    if not ai_data:
        ai_data = gemini_client.generate_educational_explanation(
            predicted_category=predicted_cat,
            confidence_score=confidence,
            normal_probability=normal_p,
            pneumonia_probability=pneumonia_p,
            model_name=model_name,
        )

    # Display Structured Explanation Cards
    st.markdown(
        f"""
        <div class="med-card" style="border-left: 4px solid {settings.COLOR_SECONDARY};">
            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #00A896;">
                {ai_data.get('title', 'AI-GENERATED EDUCATIONAL EXPLANATION')}
            </div>
            <div style="font-size: 11px; color: #64748B; margin-top: 2px;">
                Engine Source: <b>{ai_data.get('source', 'Google Gemini LLM')}</b>
            </div>
            <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 14px; margin-bottom: 6px;">Executive Summary</h4>
            <p style="font-size: 14px; color: #1E293B; line-height: 1.6;">
                {ai_data.get('executive_summary', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f"""
            <div class="med-card">
                <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 0;">🔬 Technical Probability Findings</h4>
                <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                    {ai_data.get('technical_findings', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"""
            <div class="med-card">
                <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 0;">🗺️ Grad-CAM Attention Interpretation</h4>
                <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                    {ai_data.get('gradcam_interpretation', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="med-card">
            <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 0;">⚠️ Model & Dataset Limitations</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.6;">
                <b>Architecture Constraints:</b> {ai_data.get('model_limitations', '')}
            </p>
            <p style="font-size: 13px; color: #475569; line-height: 1.6; margin-top: 8px;">
                <b>Dataset Demographics:</b> {ai_data.get('dataset_limitations', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prompt Engineering Inspection Accordion
    with st.expander("🛠️ Inspect Prompt Engineering Template & Guardrails", expanded=False):
        st.code(
            f"""
SYSTEM PROMPT SAFETY CONSTRAINTS:
1. Strict educational persona: Never provide clinical diagnoses or treatments.
2. Structured JSON response format enforced.
3. No hallucination of symptoms or patient history.

USER PROMPT STRUCTURE:
- Predicted Category: {predicted_cat}
- Confidence Score: {confidence:.4f}
- Normal Probability: {normal_p:.4f}
- Pneumonia Probability: {pneumonia_p:.4f}
- Model: {model_name}
            """,
            language="markdown",
        )
