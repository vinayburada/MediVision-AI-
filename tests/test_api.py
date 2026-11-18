"""
MediVision AI - Unit & Integration Tests for API Route Handlers
Tests route business logic, database queries, and response serialization directly.
"""
import pytest
import io
import numpy as np
import cv2
from starlette.datastructures import UploadFile
from fastapi import HTTPException

from api.main import (
    health_check,
    upload_and_validate,
    predict_image,
    generate_gradcam,
    generate_explanation,
    run_full_analysis,
    get_dataset_summary,
    get_history,
)
from api.schemas import ExplanationRequest


@pytest.fixture
def sample_png_bytes():
    """Create sample PNG file bytes for upload testing."""
    img = np.full((224, 224, 3), 120, dtype=np.uint8)
    cv2.circle(img, (112, 112), 50, (220, 220, 220), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_api_health():
    """Verify health check endpoint returns healthy status and metadata."""
    res = health_check()
    assert res.status == "healthy"
    assert res.version is not None
    assert res.database_connected is True
    assert "cpu" in res.device.lower() or "cuda" in res.device.lower()


def test_api_upload_valid(sample_png_bytes):
    """Verify upload_and_validate processes valid image."""
    uf = UploadFile(filename="test_xray.png", file=io.BytesIO(sample_png_bytes))
    res = upload_and_validate(file=uf)
    assert res.valid is True
    assert res.format == "PNG"
    assert res.original_width == 224
    assert len(res.sha256) == 64


def test_api_upload_invalid():
    """Corrupted bytes uploaded must raise HTTPException 400."""
    uf = UploadFile(filename="bad.png", file=io.BytesIO(b"bad_corrupt_data"))
    with pytest.raises(HTTPException) as exc_info:
        upload_and_validate(file=uf)
    assert exc_info.value.status_code == 400


def test_api_predict(sample_png_bytes):
    """Test model prediction endpoint."""
    uf = UploadFile(filename="test_xray.png", file=io.BytesIO(sample_png_bytes))
    res = predict_image(file=uf)
    assert res.predicted_category in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= res.confidence_score <= 1.0
    assert "NORMAL" in res.class_probabilities
    assert "PNEUMONIA" in res.class_probabilities


def test_api_gradcam(sample_png_bytes):
    """Test Grad-CAM endpoint."""
    uf = UploadFile(filename="test_xray.png", file=io.BytesIO(sample_png_bytes))
    res = generate_gradcam(file=uf, colormap="JET", alpha=0.45)
    assert res.analysis_id.startswith("MVA-")
    assert res.colormap == "JET"
    assert res.gradcam_image_base64.startswith("data:image/png;base64,")


def test_api_explain():
    """Test LLM educational explanation endpoint."""
    req = ExplanationRequest(
        predicted_category="PNEUMONIA",
        confidence_score=0.92,
        normal_probability=0.08,
        pneumonia_probability=0.92,
        model_name="DenseNet121",
    )
    res = generate_explanation(req)
    assert res.title == "AI-GENERATED EDUCATIONAL EXPLANATION"
    assert len(res.executive_summary) > 10
    assert len(res.educational_disclaimer) > 10


def test_api_full_analysis(sample_png_bytes):
    """Test full end-to-end pipeline execution in API."""
    uf = UploadFile(filename="full_test.png", file=io.BytesIO(sample_png_bytes))
    res = run_full_analysis(file=uf)
    assert res.analysis_id.startswith("MVA-")
    assert res.predicted_category in ["NORMAL", "PNEUMONIA"]
    assert res.gradcam_image_path is not None
    assert res.pdf_report_path is not None
    assert "executive_summary" in res.educational_explanation


def test_api_dataset_summary():
    """Test dataset summary endpoint."""
    res = get_dataset_summary()
    assert res.dataset_available in [True, False]
    assert "instructions" in dir(res)


def test_api_history():
    """Test history querying endpoint."""
    res = get_history(limit=5)
    assert res.total_count >= 0
    assert isinstance(res.records, list)
