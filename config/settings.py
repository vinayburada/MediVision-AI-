"""
MediVision AI - System Settings and Configuration
Comprehensive configuration module for deep learning pipelines, API, and frontend.
"""
from pathlib import Path
import os
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application-wide settings and directory paths."""

    # Project Metadata
    PROJECT_NAME: str = "MediVision AI"
    TAGLINE: str = "Intelligent Medical Image Analysis & Explainable AI Report Generation"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("APP_ENV", "development")

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    TRAIN_DATA_DIR: Path = DATA_DIR / "train"
    VAL_DATA_DIR: Path = DATA_DIR / "val"
    TEST_DATA_DIR: Path = DATA_DIR / "test"
    SAMPLES_DIR: Path = DATA_DIR / "samples"
    UPLOADS_DIR: Path = RAW_DATA_DIR / "uploads"

    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    CHECKPOINTS_DIR: Path = ARTIFACTS_DIR / "checkpoints"
    PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
    REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"

    STATIC_DIR: Path = BASE_DIR / "static"
    DATABASE_PATH: Path = BASE_DIR / "database" / "medivision.db"

    # Dataset & Classes
    CLASSES: List[str] = ["NORMAL", "PNEUMONIA"]
    NUM_CLASSES: int = 2
    CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
    IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}

    # Image Preprocessing Standards (DenseNet121 & ViT)
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    MAX_UPLOAD_SIZE_MB: int = 15

    # Deep Learning Hyperparameters
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS: int = 10
    EARLY_STOPPING_PATIENCE: int = 4
    DEFAULT_MODEL_NAME: str = "densenet121"
    BEST_MODEL_FILENAME: str = "densenet121_best.pth"
    VIT_MODEL_FILENAME: str = "vit_b16_best.pth"

    # Explainable AI (Grad-CAM)
    GRADCAM_TARGET_LAYER: str = "features.denseblock4.denselayer16.conv2"
    HEATMAP_COLORMAP: str = "JET"
    HEATMAP_ALPHA: float = 0.45

    # LLM (Google Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    LLM_TIMEOUT_SECONDS: int = 25
    LLM_MAX_RETRIES: int = 2

    # API Configuration
    API_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("APP_PORT", "8000"))
    API_PREFIX: str = "/api/v1"

    # Brand UI Theme Palette
    COLOR_PRIMARY: str = "#0F2942"      # Deep Hospital Navy
    COLOR_SECONDARY: str = "#00A896"    # Clinical Teal
    COLOR_ACCENT: str = "#028090"       # Ocean Blue Accent
    COLOR_BACKGROUND: str = "#F8FAFC"   # Modern Soft Slate
    COLOR_CARD_BG: str = "#FFFFFF"      # Crisp White Card
    COLOR_TEXT_MAIN: str = "#1E293B"    # Charcoal
    COLOR_TEXT_MUTED: str = "#64748B"   # Cool Grey
    COLOR_ALERT: str = "#E63946"        # Alert Red
    COLOR_SUCCESS: str = "#2A9D8F"      # Success Green
    COLOR_WARNING: str = "#F4A261"      # Amber Warning

    # Mandatory Legal / Educational Disclaimer
    DISCLAIMER_TEXT: str = (
        "EDUCATIONAL RESEARCH NOTICE: MediVision AI is an experimental computer vision and "
        "explainable AI research prototype designed exclusively for scientific evaluation, academic study, "
        "and educational demonstration. This software is NOT a certified medical device and has NOT been "
        "cleared or approved by the FDA, CE, CDSCO, or any international medical regulatory authority. "
        "The generated predictions, probability scores, Grad-CAM heatmaps, and AI summaries do NOT "
        "constitute a medical diagnosis, clinical opinion, or treatment recommendation. Always consult "
        "a qualified, licensed radiologist or healthcare provider for any diagnostic or patient care decisions."
    )

    def ensure_directories(self) -> None:
        """Create all required working directories if they do not already exist."""
        dirs = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.TRAIN_DATA_DIR,
            self.VAL_DATA_DIR,
            self.TEST_DATA_DIR,
            self.SAMPLES_DIR,
            self.UPLOADS_DIR,
            self.CHECKPOINTS_DIR,
            self.PLOTS_DIR,
            self.REPORTS_DIR,
            self.STATIC_DIR,
            self.BASE_DIR / "database",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
