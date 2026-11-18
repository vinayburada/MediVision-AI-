"""
MediVision AI - Unit Tests for ReportLab PDF Report Generation
"""
import pytest
from pathlib import Path
import tempfile
import numpy as np
import cv2

from utils.pdf_generator import PDFReportGenerator
from config.settings import settings


@pytest.fixture
def temp_images_and_pdf_dir():
    """Create temp directory and dummy images for PDF generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)

        # Save dummy original image
        orig_path = dir_path / "orig.png"
        img1 = np.full((200, 200, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(orig_path), img1)

        # Save dummy gradcam image
        grad_path = dir_path / "grad.png"
        img2 = np.full((200, 200, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(grad_path), img2)

        yield dir_path, orig_path, grad_path


def test_pdf_report_generation(temp_images_and_pdf_dir):
    """Test generating full PDF document with ReportLab."""
    pdf_dir, orig_path, grad_path = temp_images_and_pdf_dir

    generator = PDFReportGenerator(output_dir=pdf_dir)

    pred_data = {
        "predicted_category": "PNEUMONIA",
        "confidence_score": 0.934,
        "class_probabilities": {"NORMAL": 0.066, "PNEUMONIA": 0.934},
        "inference_time_ms": 38.2,
        "model_name": "DenseNet121",
    }

    ai_exp = {
        "executive_summary": "Test executive summary content.",
        "technical_findings": "Test technical findings.",
        "gradcam_interpretation": "Test spatial hotspot interpretation.",
        "model_limitations": "Test model limitations.",
        "dataset_limitations": "Test dataset limitations.",
    }

    pdf_path = generator.generate_report(
        analysis_id="MVA-TEST-PDF-001",
        original_image_path=orig_path,
        gradcam_image_path=grad_path,
        prediction_data=pred_data,
        ai_explanation=ai_exp,
    )

    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 1000  # Non-trivial PDF file size
    assert str(pdf_path).endswith(".pdf")
