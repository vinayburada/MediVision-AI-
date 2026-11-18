"""
MediVision AI - Page 6: About Project & Scientific Documentation
System architecture, model designs, data leakage prevention, and bioethical guidelines.
"""
import streamlit as st
from config.settings import settings
from utils.ui_components import apply_custom_css, render_sidebar, render_disclaimer_banner

st.set_page_config(
    page_title="About Project | MediVision AI",
    page_icon="📖",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.title("📖 About MediVision AI")
st.markdown("Scientific documentation, deep learning system architecture, and clinical ethical boundaries.")
render_disclaimer_banner()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Objectives & Mission",
    "🛠️ Technology Stack",
    "🧠 Architectures & Grad-CAM",
    "📁 Dataset & Patient Leakage",
    "⚖️ Bioethics & Disclaimers",
])

with tab1:
    st.markdown("### Project Objectives")
    st.markdown(
        f"""
        <div class="med-card">
            <h4 style="color: {settings.COLOR_PRIMARY}; margin-top: 0;">Educational AI Mission</h4>
            <p style="font-size: 13.5px; color: #334155; line-height: 1.6;">
                <b>MediVision AI</b> was conceived and engineered to bridge the interpretability gap between black-box
                deep learning computer vision models and clinical radiography education.
            </p>
            <p style="font-size: 13.5px; color: #334155; line-height: 1.6;">
                Standard Convolutional Neural Networks achieve high classification accuracy on benchmarks, but often rely on
                spurious background correlations (such as hospital acquisition markers, radiograph contrast differences, or patient age artifacts).
                MediVision AI tackles this challenge through:
            </p>
            <ul style="font-size: 13px; color: #475569; line-height: 1.6;">
                <li><b>Transparent Spatial Grounding:</b> Computing Grad-CAM heatmaps at the penultimate convolutional block.</li>
                <li><b>Generative AI Synthesis:</b> Using Google Gemini LLM to interpret probabilities, explaining what the model saw and did not see.</li>
                <li><b>Patient-Level Leakage Prevention:</b> Grouping multiple radiographs by patient ID before splitting to prevent overfitted optimistic validation.</li>
                <li><b>Strict Non-Diagnostic Guardrails:</b> Rejecting diagnostic claims or clinical prescriptions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("### Complete Technology Stack")
    st.markdown(
        """
        <div class="med-card">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr style="border-bottom: 2px solid #E2E8F0; text-align: left;">
                    <th style="padding: 8px;">Domain</th>
                    <th style="padding: 8px;">Technology</th>
                    <th style="padding: 8px;">Architectural Purpose</th>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Deep Learning</b></td>
                    <td style="padding: 8px;">PyTorch, Torchvision</td>
                    <td style="padding: 8px;">DenseNet-121 CNN & Vision Transformer (ViT-B/16)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Computer Vision</b></td>
                    <td style="padding: 8px;">OpenCV, Pillow</td>
                    <td style="padding: 8px;">Bicubic resizing, ImageNet standardization, corrupted file detection</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Explainable AI (XAI)</b></td>
                    <td style="padding: 8px;">PyTorch Backward Hooks, Grad-CAM</td>
                    <td style="padding: 8px;">Activation heatmap extraction and alpha-blended colormaps</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Generative AI</b></td>
                    <td style="padding: 8px;">Google Gemini API, JSON Prompts</td>
                    <td style="padding: 8px;">Structured educational explanations and scientific summarization</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Backend REST API</b></td>
                    <td style="padding: 8px;">FastAPI, Uvicorn, Pydantic</td>
                    <td style="padding: 8px;">Modular microservices, validation schemas, file upload security</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Interactive UI</b></td>
                    <td style="padding: 8px;">Streamlit, Plotly</td>
                    <td style="padding: 8px;">Medical grade multi-page dashboard, responsive cards, interactive charts</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Database</b></td>
                    <td style="padding: 8px;">SQLite 3</td>
                    <td style="padding: 8px;">Parameterized audit trails, search indexing, deduplication</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px;"><b>Reporting</b></td>
                    <td style="padding: 8px;">ReportLab Platypus</td>
                    <td style="padding: 8px;">Publication-quality clinical research PDF reports with two-pass canvas</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>Quality Assurance</b></td>
                    <td style="padding: 8px;">Pytest, Pytest-Asyncio, HTTPX</td>
                    <td style="padding: 8px;">End-to-end unit, integration, and endpoint testing</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown("### Neural Architectures & Grad-CAM Mechanics")
    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F2942; margin-top: 0;">DenseNet-121 Architecture</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                DenseNet connects all layers directly with each other (within dense blocks). Each layer receives feature-maps
                of all preceding layers as inputs: \\(x_l = H_l([x_0, x_1, ..., x_{{l-1}}])\\).
                This encourages heavy feature reuse, reduces parameter count compared to ResNet-50, and creates strong gradient propagation.
            </p>
            <h4 style="color: #0F2942; margin-top: 14px;">Vision Transformer (ViT-B/16)</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                ViT segments 224×224 images into 196 non-overlapping 16×16 patches, projects each patch into a 768-dimensional token,
                adds 1D learnable position embeddings, and passes tokens through 12 multi-head self-attention Transformer encoders.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab4:
    st.markdown("### Dataset Sourcing & Patient Leakage Prevention")
    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F2942; margin-top: 0;">Dataset Provenance</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                The dataset used is the <b>Chest X-Ray Images (Pneumonia)</b> dataset curated by Paul Mooney, sourced from Guangzhou
                Women and Children’s Medical Center. It contains 5,856 pediatric radiographs classified into <code>NORMAL</code> and <code>PNEUMONIA</code>.
            </p>
            <h4 style="color: #0F2942; margin-top: 12px;">Preventing Patient-Level Data Leakage</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                In medical image datasets, multiple radiographs (taken at different angles or days) often originate from the same patient.
                If random image-level splitting is applied, images of Patient A appear in both training and test sets.
                The network may memorize patient-specific anatomical characteristics (such as thoracic shape or bone density) rather than
                pathology, producing falsely inflated test accuracy.
            </p>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                <b>MediVision AI's Solution:</b> The <code>DatasetManager</code> parses patient identifiers (e.g. <code>person1946</code>)
                from filenames and guarantees that all radiographs belonging to a patient are strictly assigned to either train, validation, or test.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab5:
    st.markdown("### Bioethics & Regulatory Disclaimers")
    st.markdown(
        f"""
        <div class="med-card" style="border-left: 4px solid #E11D48;">
            <h4 style="color: #9F1239; margin-top: 0;">Bioethical Research Boundaries</h4>
            <p style="font-size: 13px; color: #4C0519; line-height: 1.6;">
                1. <b>Not a Medical Diagnostic Device:</b> This software is not cleared by the FDA (510k/PMA), CE mark, CDSCO, or MHRA.
                It must never be utilized in clinical workflows, emergency triage, or patient treatment pathways.
            </p>
            <p style="font-size: 13px; color: #4C0519; line-height: 1.6;">
                2. <b>HIPAA & Privacy:</b> No Protected Health Information (PHI) such as patient names, medical record numbers (MRNs),
                or hospital barcodes are stored in MediVision AI's SQLite database. Images are anonymized and tracked via SHA-256 hashes.
            </p>
            <p style="font-size: 13px; color: #4C0519; line-height: 1.6;">
                3. <b>Demographic & Acquisition Bias:</b> The primary academic dataset comprises pediatric patients (1–5 years old).
                Predictions applied to adult chest radiographs will exhibit out-of-distribution domain shift.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
