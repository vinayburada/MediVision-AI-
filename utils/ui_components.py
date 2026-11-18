"""
MediVision AI - Streamlit UI Design System & Component Library
Provides clinical brand styling, CSS injection, responsive metric cards, and disclaimer banners.
"""
import streamlit as st
from config.settings import settings


def apply_custom_css():
    """Injects high-end medical dashboard typography, card styles, and color palette."""
    css = f"""
    <style>
        /* Base typography & palette */
        :root {{
            --primary: {settings.COLOR_PRIMARY};
            --secondary: {settings.COLOR_SECONDARY};
            --accent: {settings.COLOR_ACCENT};
            --bg: {settings.COLOR_BACKGROUND};
            --card-bg: {settings.COLOR_CARD_BG};
            --text-main: {settings.COLOR_TEXT_MAIN};
            --text-muted: {settings.COLOR_TEXT_MUTED};
        }}

        /* App background */
        .stApp {{
            background-color: #F8FAFC;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        /* Card container */
        .med-card {{
            background: #FFFFFF;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px 0 rgba(0, 0, 0, 0.04);
            border: 1px solid #E2E8F0;
            margin-bottom: 18px;
        }}

        /* Header hero container */
        .hero-banner {{
            background: linear-gradient(135deg, {settings.COLOR_PRIMARY} 0%, #1A365D 100%);
            color: #FFFFFF;
            border-radius: 14px;
            padding: 26px 32px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(15, 41, 66, 0.15);
        }}

        .hero-title {{
            font-size: 26px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 6px;
        }}

        .hero-subtitle {{
            font-size: 14px;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.5;
        }}

        /* Badge styles */
        .badge-normal {{
            background-color: #DEF7EC;
            color: #03543F;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 13px;
            display: inline-block;
        }}

        .badge-pneumonia {{
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 13px;
            display: inline-block;
        }}

        /* Metric card */
        .metric-box {{
            background: #FFFFFF;
            border-radius: 10px;
            padding: 16px;
            border-left: 4px solid {settings.COLOR_SECONDARY};
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
            color: #64748B;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: #0F2942;
            margin-top: 4px;
        }}

        /* Educational disclaimer banner */
        .disclaimer-banner {{
            background: #FFF1F2;
            border-left: 4px solid #E11D48;
            padding: 12px 18px;
            border-radius: 6px;
            margin: 16px 0;
            font-size: 12.5px;
            color: #9F1239;
            line-height: 1.5;
        }}

        /* Buttons */
        .stButton>button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            border-color: {settings.COLOR_SECONDARY};
            color: {settings.COLOR_SECONDARY};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar():
    """Renders persistent branding and system health monitor in Streamlit sidebar."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 15px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 15px;">
                <h2 style="color: {settings.COLOR_PRIMARY}; margin: 0; font-size: 22px;">🩺 MediVision AI</h2>
                <p style="color: #64748B; font-size: 11px; margin: 4px 0 0 0;">Intelligent Medical Vision & Explainability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ System Status")

        # PyTorch & Device status
        import torch
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.caption(f"**Compute Device:** `{device_str}`")
        st.caption(f"**PyTorch Version:** `{torch.__version__}`")

        # Model Checkpoint status
        from pathlib import Path
        best_ckpt = settings.CHECKPOINTS_DIR / settings.BEST_MODEL_FILENAME
        if best_ckpt.exists():
            st.success("DenseNet121: Checkpoint Loaded", icon="✅")
        else:
            st.info("DenseNet121: Backbone Active", icon="ℹ️")

        # LLM Status
        from llm.gemini_client import gemini_client
        if gemini_client.is_configured:
            st.success("Google Gemini: Connected", icon="✨")
        else:
            st.warning("Google Gemini: Offline Mode (Key Not Set)", icon="⚠️")

        # Database Status
        from database.db import db
        st.caption(f"**Database:** `SQLite ({db.db_path.name})`")

        st.divider()

        # Mini Disclaimer in sidebar
        st.markdown(
            """
            <div style="font-size: 10.5px; color: #94A3B8; line-height: 1.4;">
                <b>Notice:</b> Educational research prototype. Not certified for clinical medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_disclaimer_banner():
    """Renders standardized educational research disclaimer."""
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <b>⚠️ EDUCATIONAL RESEARCH NOTICE:</b> {settings.DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )
