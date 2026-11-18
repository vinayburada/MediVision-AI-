"""
MediVision AI - Page 1: Image Analysis
Interactive chest radiograph upload, validation, classification, Grad-CAM, and PDF generation.
"""
from pathlib import Path
import io
import time
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner
from preprocessing.image_preprocessing import (
    ImagePreprocessor,
    default_preprocessor,
    ImageValidationError,
)
from preprocessing.image_loader import DatasetManager
from models.predict import Predictor
from explainability.gradcam import GradCAM
from llm.report_generator import report_generator
from database.db import db
from utils.pdf_generator import pdf_generator
from utils.helpers import generate_analysis_id, sanitize_filename

st.set_page_config(
    page_title="Image Analysis | MediVision AI",
    page_icon="📸",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.title("📸 Chest Radiograph Classification & Analysis")
st.markdown("Upload a frontal chest X-ray or select a pre-loaded educational sample phantom.")
render_disclaimer_banner()

# Initialize DatasetManager to ensure samples exist
dm = DatasetManager()
sample_files = dm.generate_synthetic_educational_samples(count_per_class=4)

# Step 1: Input Selection (Upload vs Sample Selector)
input_mode = st.radio("Select Input Mode:", ["Upload Image File", "Select Pre-loaded Educational Sample"], horizontal=True)

selected_image_bytes = None
image_filename = "sample.png"

if input_mode == "Upload Image File":
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPEG, PNG, TIFF):",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help=f"Maximum allowed file size: {settings.MAX_UPLOAD_SIZE_MB}MB",
    )
    if uploaded_file is not None:
        selected_image_bytes = uploaded_file.read()
        image_filename = uploaded_file.name
else:
    sample_options = {}
    for p_str in sample_files:
        p = Path(p_str)
        sample_options[f"{p.parent.name} - {p.name}"] = p_str

    chosen_key = st.selectbox("Choose a sample chest X-ray:", list(sample_options.keys()))
    if chosen_key:
        chosen_path = Path(sample_options[chosen_key])
        with open(chosen_path, "rb") as f:
            selected_image_bytes = f.read()
        image_filename = chosen_path.name

# Model Architecture Selector
col_model, col_thresh = st.columns([2, 1])
with col_model:
    model_choice = st.selectbox(
        "Select Deep Learning Model Architecture:",
        ["DenseNet-121 (Primary CNN)", "Vision Transformer (ViT-B/16)"],
        index=0,
    )
    chosen_model_name = "densenet121" if "DenseNet" in model_choice else "vit_b_16"

with col_thresh:
    cam_colormap = st.selectbox("Grad-CAM Colormap:", ["JET", "VIRIDIS", "TURBO", "PLASMA"], index=0)

if selected_image_bytes:
    # Validate Image
    try:
        val_meta = ImagePreprocessor.validate_image_file(selected_image_bytes)
        is_valid = True
    except ImageValidationError as e:
        st.error(f"Image Validation Error: {e}")
        is_valid = False

    if is_valid:
        st.write("")
        col_preview, col_action = st.columns([1, 1.5])

        with col_preview:
            st.markdown("#### Input Radiograph Preview")
            pil_img = Image.open(io.BytesIO(selected_image_bytes))
            st.image(pil_img, use_container_width=True, caption=f"{image_filename} ({val_meta['original_width']}×{val_meta['original_height']} px)")

            # Image metadata pills
            st.caption(f"**Format:** `{val_meta['format']}` | **Size:** `{val_meta['file_size_kb']} KB` | **Channels:** `{val_meta['channels']}`")

        with col_action:
            st.markdown("#### Run Clinical AI Analysis")
            st.info("The image will be resized to 224×224, normalized via ImageNet parameters, and passed through the classification head.")

            analyze_btn = st.button("🚀 Analyze Chest Radiograph", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner("Processing image and executing model forward pass..."):
                analysis_id = generate_analysis_id()
                safe_name = sanitize_filename(image_filename)

                # Save raw upload
                raw_path = settings.UPLOADS_DIR / f"{analysis_id}_{safe_name}"
                with open(raw_path, "wb") as f:
                    f.write(selected_image_bytes)

                # Initialize Predictor
                predictor = Predictor(model_name=chosen_model_name)
                pred_result = predictor.predict(selected_image_bytes)

                predicted_cat = pred_result["predicted_category"]
                confidence = pred_result["confidence_score"]
                class_probs = pred_result["class_probabilities"]
                latency = pred_result["inference_time_ms"]

                # Generate Grad-CAM
                gradcam_img_path = None
                overlay_rgb = None
                heatmap_2d = None

                try:
                    cam = GradCAM(predictor.model)
                    heatmap_2d = cam.generate_heatmap(pred_result["input_tensor"], target_class_idx=pred_result["predicted_index"])
                    _, overlay_rgb = cam.overlay_heatmap(pred_result["original_rgb"], heatmap_2d, colormap=cam_colormap)
                    cam.remove_hooks()

                    gradcam_img_path = settings.PLOTS_DIR / f"gradcam_{analysis_id}.png"
                    cv2.imwrite(str(gradcam_img_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                except Exception as cam_err:
                    st.warning(f"Grad-CAM note: {cam_err}")

                # Generate AI Explanation
                explanation_pack = report_generator.generate_full_explanation(pred_result, heatmap_2d)
                ai_explanation = explanation_pack["educational_explanation"]

                # Generate PDF Report
                pdf_path = pdf_generator.generate_report(
                    analysis_id=analysis_id,
                    original_image_path=raw_path,
                    gradcam_image_path=gradcam_img_path,
                    prediction_data=pred_result,
                    ai_explanation=ai_explanation,
                )

                # Save record to SQLite
                db.save_analysis({
                    "analysis_id": analysis_id,
                    "image_hash": val_meta["sha256"],
                    "image_filename": safe_name,
                    "image_path": str(raw_path),
                    "predicted_category": predicted_cat,
                    "confidence_score": confidence,
                    "normal_probability": class_probs["NORMAL"],
                    "pneumonia_probability": class_probs["PNEUMONIA"],
                    "model_name": chosen_model_name,
                    "model_version": settings.VERSION,
                    "inference_time_ms": latency,
                    "gradcam_image_path": str(gradcam_img_path) if gradcam_img_path else "",
                    "pdf_report_path": str(pdf_path),
                    "ai_explanation_json": ai_explanation,
                    "notes": "Generated from Streamlit Image Analysis",
                })

                # Store current analysis in session state for cross-page persistence
                st.session_state["current_analysis"] = {
                    "analysis_id": analysis_id,
                    "predicted_category": predicted_cat,
                    "confidence_score": confidence,
                    "class_probabilities": class_probs,
                    "inference_time_ms": latency,
                    "model_name": chosen_model_name,
                    "raw_image_path": str(raw_path),
                    "gradcam_image_path": str(gradcam_img_path) if gradcam_img_path else None,
                    "pdf_report_path": str(pdf_path),
                    "ai_explanation": ai_explanation,
                }

        # Step 2: Display Results if available in session state
        if "current_analysis" in st.session_state:
            res = st.session_state["current_analysis"]
            st.divider()
            st.markdown(f"### 📋 Analysis Results (`{res['analysis_id']}`)")

            # Prediction Card & Metrics
            pred_col1, pred_col2, pred_col3 = st.columns([1.5, 1, 1])

            cat = res["predicted_category"]
            badge_class = "badge-pneumonia" if cat == "PNEUMONIA" else "badge-normal"

            with pred_col1:
                st.markdown(
                    f"""
                    <div class="med-card">
                        <div style="font-size: 13px; color: #64748B; font-weight: 600;">PREDICTED CATEGORY</div>
                        <div style="margin: 8px 0;">
                            <span class="{badge_class}" style="font-size: 20px; padding: 6px 18px;">{cat}</span>
                        </div>
                        <div style="font-size: 12px; color: #64748B;">Architecture: <b>{res['model_name']}</b> | Latency: <b>{res['inference_time_ms']:.1f} ms</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with pred_col2:
                st.metric("Normal Probability", f"{res['class_probabilities']['NORMAL'] * 100:.2f}%")
                st.progress(res['class_probabilities']['NORMAL'])

            with pred_col3:
                st.metric("Pneumonia Probability", f"{res['class_probabilities']['PNEUMONIA'] * 100:.2f}%")
                st.progress(res['class_probabilities']['PNEUMONIA'])

            # Side-by-side Images
            st.markdown("#### 🔬 Visual Comparison & Attention Heatmap")
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.image(res["raw_image_path"], caption="Input Chest Radiograph", use_container_width=True)

            with img_col2:
                if res.get("gradcam_image_path") and Path(res["gradcam_image_path"]).exists():
                    st.image(res["gradcam_image_path"], caption="Grad-CAM Penultimate Layer Attention Overlay", use_container_width=True)
                else:
                    st.info("Grad-CAM visualization not available for this run.")

            # AI Explanation Accordion
            with st.expander("✨ View AI-Generated Educational Summary", expanded=True):
                ai_exp = res["ai_explanation"]
                st.caption(f"**Source:** `{ai_exp.get('source', 'MediVision AI')}`")
                st.markdown(f"**Executive Summary:** {ai_exp.get('executive_summary', '')}")
                st.markdown(f"**Technical Findings:** {ai_exp.get('technical_findings', '')}")
                st.markdown(f"**Grad-CAM Spatial Interpretation:** {ai_exp.get('gradcam_interpretation', '')}")
                st.markdown(f"**Limitations:** {ai_exp.get('model_limitations', '')}")

            # PDF Download Button
            pdf_path_obj = Path(res["pdf_report_path"])
            if pdf_path_obj.exists():
                with open(pdf_path_obj, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Research PDF Report",
                        data=pdf_file.read(),
                        file_name=f"MediVision_Report_{res['analysis_id']}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
