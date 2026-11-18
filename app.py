"""
MediVision AI - Home Dashboard
Main entry point for the Streamlit medical AI application.
"""
from pathlib import Path
import streamlit as st
import pandas as pd

from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner
from database.db import db
from dashboard.analytics import analytics_engine
from preprocessing.image_loader import DatasetManager

# Page configuration
st.set_page_config(
    page_title=f"{settings.PROJECT_NAME} | Clinical Research AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar()

# Hero Banner
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">🩺 {settings.PROJECT_NAME}</div>
        <div class="hero-subtitle">{settings.TAGLINE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_disclaimer_banner()

# Step 1: Summary KPI Metrics
kpis = analytics_engine.get_summary_kpis()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">{kpis['total_analyses']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Pneumonia Positive Rate</div>
            <div class="metric-value" style="color: {settings.COLOR_ALERT};">{kpis['positive_rate']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Mean Model Confidence</div>
            <div class="metric-value">{kpis['avg_confidence']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Average Latency</div>
            <div class="metric-value">{kpis['avg_latency_ms']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Step 2: Main Layout (Project Overview & Quick Actions vs Dataset & Models)
left_col, right_col = st.columns([1.8, 1.2])

with left_col:
    st.markdown("### 🔬 System Overview & Capabilities")
    st.markdown(
        """
        <div class="med-card">
            <p style="font-size: 14px; color: #334155; line-height: 1.6;">
                <b>MediVision AI</b> integrates transfer learning with Explainable Artificial Intelligence (XAI)
                and Large Language Models to analyze anterior-posterior chest radiography. The platform demonstrates:
            </p>
            <ul style="font-size: 13.5px; color: #475569; line-height: 1.6;">
                <li><b>Deep Learning Classification:</b> DenseNet121 architecture with dense feature reuse for distinguishing <code>NORMAL</code> vs <code>PNEUMONIA</code>.</li>
                <li><b>Vision Transformer (ViT-B/16):</b> Attention-based architecture comparing patch self-attention to convolutional feature maps.</li>
                <li><b>Visual Explainability:</b> Gradient-weighted Class Activation Mapping (Grad-CAM) highlighting spatial activations.</li>
                <li><b>Generative AI Reporting:</b> Google Gemini API synthesizes technical metrics into structured educational findings.</li>
                <li><b>Clinical Research PDF:</b> Downloadable publications with side-by-side radiograph overlays and audit timestamps.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Cards
    st.markdown("### 🚀 Quick Navigation")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.markdown(
            f"""
            <div class="med-card" style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 6px;">📸</div>
                <div style="font-weight: 700; color: {settings.COLOR_PRIMARY};">Image Analysis</div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Upload & classify chest radiographs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav2:
        st.markdown(
            f"""
            <div class="med-card" style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 6px;">🔍</div>
                <div style="font-weight: 700; color: {settings.COLOR_PRIMARY};">Explainable AI</div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Grad-CAM heatmaps & layer analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav3:
        st.markdown(
            f"""
            <div class="med-card" style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 6px;">📊</div>
                <div style="font-weight: 700; color: {settings.COLOR_PRIMARY};">Analytics</div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Confusion matrix & ROC metrics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    # Model Status Card
    st.markdown("### 🧠 Model Architecture")
    best_ckpt = settings.CHECKPOINTS_DIR / settings.BEST_MODEL_FILENAME
    ckpt_status = f"Trained Checkpoint Available ({best_ckpt.name})" if best_ckpt.exists() else "Pretrained Backbone (Ready for Fine-Tuning)"
    st.markdown(
        f"""
        <div class="med-card">
            <div style="font-weight: 700; color: {settings.COLOR_PRIMARY}; font-size: 15px;">DenseNet-121 (Primary Model)</div>
            <div style="font-size: 12px; color: #00A896; font-weight: 600; margin-top: 2px;">{ckpt_status}</div>
            <div style="font-size: 13px; color: #64748B; margin-top: 8px;">
                <b>Parameters:</b> 7.05 Million<br>
                <b>Target Layer:</b> <code>denseblock4.denselayer16.conv2</code><br>
                <b>Input Dimensions:</b> 224 × 224 × 3 (ImageNet Normalized)<br>
                <b>Alternative:</b> Vision Transformer (ViT-B/16)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Status Card
    st.markdown("### 📁 Dataset Status")
    dm = DatasetManager()
    summary = dm.get_dataset_summary()
    has_data = summary["dataset_available"]

    if has_data:
        st.markdown(
            f"""
            <div class="med-card">
                <div style="font-weight: 700; color: {settings.COLOR_PRIMARY};">Chest X-Ray (Pneumonia) Dataset</div>
                <div style="font-size: 13px; color: #334155; margin-top: 6px;">
                    <b>Total Images:</b> {summary['grand_total']}<br>
                    <b>Normal:</b> {summary['class_distribution'].get('NORMAL', 0)} | 
                    <b>Pneumonia:</b> {summary['class_distribution'].get('PNEUMONIA', 0)}<br>
                    <b>Class Ratio:</b> {summary.get('imbalance_ratio', 'N/A')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="med-card" style="border-left: 4px solid #F59E0B;">
                <div style="font-weight: 700; color: #B45309;">Kaggle Dataset Not Detected</div>
                <p style="font-size: 12px; color: #78350F; margin-top: 4px; line-height: 1.4;">
                    Built-in synthetic educational chest radiograph phantoms are activated for out-of-the-box demonstration.
                    To train on the full clinical cohort, download the dataset from Kaggle.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Step 3: Recent Analyses Table
st.markdown("### 📋 Recent Analyses Audit Log")
recent_analyses = db.list_analyses(limit=5)
if recent_analyses:
    recent_df = pd.DataFrame(recent_analyses)[[
        "analysis_id", "image_filename", "predicted_category",
        "confidence_score", "inference_time_ms", "analysis_timestamp"
    ]]
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
else:
    st.info("No analyses logged yet. Head over to **1_Image_Analysis** to upload and evaluate your first chest radiograph.")
