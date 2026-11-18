"""
MediVision AI - Page 2: Explainable AI (Grad-CAM Deep Dive)
Interactive spatial feature inspection, layer activations, colormap adjustment, and XAI education.
"""
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner
from preprocessing.image_loader import DatasetManager
from models.predict import Predictor
from explainability.gradcam import GradCAM

st.set_page_config(
    page_title="Explainable AI | MediVision AI",
    page_icon="🔍",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.title("🔍 Explainable AI – Grad-CAM Spatial Analysis")
st.markdown("Inspect how deep convolutional layers formulate predictions through gradient-weighted activation mapping.")
render_disclaimer_banner()

# Load samples or session image
dm = DatasetManager()
sample_files = dm.generate_synthetic_educational_samples(count_per_class=4)

# Determine image source
st.markdown("### 1. Select Radiograph for Attention Inspection")
img_source = st.radio("Source:", ["Use Current Analysis from Image Analysis Page", "Select from Educational Samples"], horizontal=True)

target_image_path = None
if img_source == "Use Current Analysis from Image Analysis Page" and "current_analysis" in st.session_state:
    target_image_path = st.session_state["current_analysis"].get("raw_image_path")
    if not target_image_path or not Path(target_image_path).exists():
        target_image_path = None

if not target_image_path:
    sample_dict = {f"{Path(p).parent.name} - {Path(p).name}": p for p in sample_files}
    chosen = st.selectbox("Educational Chest Radiograph Phantoms:", list(sample_dict.keys()))
    target_image_path = sample_dict[chosen]

# Interactive Parameter Controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    colormap_choice = st.selectbox("Heatmap Colormap Palette:", ["JET", "VIRIDIS", "TURBO", "PLASMA", "MAGMA"], index=0)
with col_ctrl2:
    alpha_val = st.slider("Heatmap Blending Opacity (α):", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
with col_ctrl3:
    target_class_choice = st.selectbox("Grad-CAM Target Category:", ["Highest Probability Class", "NORMAL", "PNEUMONIA"])

if target_image_path and Path(target_image_path).exists():
    predictor = Predictor(model_name="densenet121")
    pred_res = predictor.predict(target_image_path)

    # Determine class index
    if target_class_choice == "NORMAL":
        target_idx = 0
    elif target_class_choice == "PNEUMONIA":
        target_idx = 1
    else:
        target_idx = pred_res["predicted_index"]

    # Compute Grad-CAM
    cam = GradCAM(predictor.model)
    try:
        heatmap_2d = cam.generate_heatmap(pred_res["input_tensor"], target_class_idx=target_idx)
        colored_rgb, overlay_rgb = cam.overlay_heatmap(
            pred_res["original_rgb"], heatmap_2d, alpha=alpha_val, colormap=colormap_choice
        )
    finally:
        cam.remove_hooks()

    # Step 2: Three-Way Visual Comparison
    st.markdown("### 2. Side-by-Side Visual Inspection")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Input X-Ray")
        st.image(pred_res["original_rgb"], caption="Preprocessed Radiograph (224×224)", use_container_width=True)

    with c2:
        st.markdown(f"#### Activation Heatmap ({colormap_choice})")
        st.image(colored_rgb, caption=f"Grad-CAM Feature Heatmap (Class: {settings.CLASSES[target_idx]})", use_container_width=True)

    with c3:
        st.markdown("#### Blended Overlay")
        st.image(overlay_rgb, caption=f"Alpha Blended Overlay (α = {alpha_val})", use_container_width=True)

    # Download heatmap
    _, encoded_buffer = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    st.download_button(
        label="📥 Download High-Resolution Grad-CAM Overlay",
        data=encoded_buffer.tobytes(),
        file_name=f"gradcam_{Path(target_image_path).stem}_{colormap_choice}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.divider()

    # Step 3: Deep Technical Explanation of Grad-CAM
    st.markdown("### 3. Mathematical Mechanism of Grad-CAM")
    exp_info = GradCAM.explain_gradcam_educational()

    st.markdown(
        f"""
        <div class="med-card">
            <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 0;">How Grad-CAM Computes Spatial Attention</h4>
            <p style="font-size: 13.5px; color: #334155; line-height: 1.6;">
                1. <b>Forward Propagation:</b> The input radiograph passes through DenseNet-121's convolutional dense blocks,
                producing 1,024 feature activation maps \\(A^k\\) at the penultimate layer (<code>denseblock4.denselayer16.conv2</code>).
            </p>
            <p style="font-size: 13.5px; color: #334155; line-height: 1.6;">
                2. <b>Gradient Backpropagation:</b> Gradients of the target category score \\(y^c\\) with respect to each activation map
                \\(A^k\\) are calculated:
            </p>
            <div style="background: #F1F5F9; padding: 10px 16px; border-radius: 6px; font-family: monospace; font-size: 13px; margin: 8px 0;">
                α_k^c = (1 / Z) * ∑_i ∑_j (∂y^c / ∂A_{{i,j}}^k)
            </div>
            <p style="font-size: 13.5px; color: #334155; line-height: 1.6;">
                3. <b>Global Average Pooling & Linear Combination:</b> Importance weights \\(α_k^c\\) are pooled spatially, weighting each feature map.
                A Rectified Linear Unit (<b>ReLU</b>) filter is applied to retain only features with a positive influence on the target class:
            </p>
            <div style="background: #F1F5F9; padding: 10px 16px; border-radius: 6px; font-family: monospace; font-size: 13px; margin: 8px 0;">
                L_{{Grad-CAM}}^c = ReLU( ∑_k α_k^c * A^k )
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="med-card" style="border-left: 4px solid #E11D48;">
            <h4 style="color: #9F1239; margin-top: 0;">⚠️ Crucial Scientific & Clinical Limitations of Grad-CAM</h4>
            <ul style="font-size: 13px; color: #4C0519; line-height: 1.6;">
                <li><b>Correlation, Not Pathology:</b> Grad-CAM reflects pixel groupings that maximized neural layer weights, which may correlate with non-pathological features such as patient bone density, surgical staples, ECG leads, or image contrast artifacts.</li>
                <li><b>Resolution Constraints:</b> Feature maps at the 4th dense block are downsampled (7×7 spatial grid), requiring bicubic upsampling back to 224×224. Spatial boundaries are therefore smooth approximations, not precise anatomical margins.</li>
                <li><b>Not Clinical Reasoning:</b> High activation in lung parenchyma does NOT prove infection etiology or exclude alternative pathologies such as pulmonary edema, atelectasis, or pleural effusion.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
