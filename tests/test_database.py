"""
MediVision AI - Unit Tests for SQLite Database & History
"""
import pytest
from pathlib import Path
import tempfile

from database.db import DatabaseManager
from config.settings import settings


@pytest.fixture
def test_db():
    """Create isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    db_mgr = DatabaseManager(db_path=db_path)
    yield db_mgr
    # Cleanup
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass


def test_db_init_and_save(test_db):
    """Test schema creation and saving a valid analysis record."""
    sample_record = {
        "analysis_id": "MVA-TEST-001",
        "image_hash": "a" * 64,
        "image_filename": "test_xray.png",
        "image_path": "/data/test_xray.png",
        "predicted_category": "PNEUMONIA",
        "confidence_score": 0.9421,
        "normal_probability": 0.0579,
        "pneumonia_probability": 0.9421,
        "model_name": "densenet121",
        "model_version": "1.0.0",
        "inference_time_ms": 42.5,
        "analysis_timestamp": "2026-11-18T12:00:00Z",
        "gradcam_image_path": "/artifacts/plots/gradcam_test.png",
        "pdf_report_path": "/artifacts/reports/report_test.pdf",
        "ai_explanation_json": {"summary": "Test summary"},
        "notes": "Automated unit test record",
    }

    res_id = test_db.save_analysis(sample_record)
    assert res_id == "MVA-TEST-001"

    # Retrieve record
    retrieved = test_db.get_analysis_by_id("MVA-TEST-001")
    assert retrieved is not None
    assert retrieved["predicted_category"] == "PNEUMONIA"
    assert retrieved["confidence_score"] == 0.9421
    assert retrieved["ai_explanation_parsed"] == {"summary": "Test summary"}


def test_list_and_filter(test_db):
    """Test filtering analyses by category and text search."""
    test_db.save_analysis({
        "analysis_id": "MVA-NORM-100",
        "image_hash": "b" * 64,
        "image_filename": "normal_01.png",
        "image_path": "/path/normal_01.png",
        "predicted_category": "NORMAL",
        "confidence_score": 0.88,
        "normal_probability": 0.88,
        "pneumonia_probability": 0.12,
        "model_name": "densenet121",
        "model_version": "1.0.0",
        "inference_time_ms": 30.0,
        "analysis_timestamp": "2026-11-18T10:00:00Z",
    })

    # Search by category
    results = test_db.list_analyses(category="NORMAL")
    assert len(results) >= 1
    assert results[0]["analysis_id"] == "MVA-NORM-100"

    # Search by query
    results_q = test_db.list_analyses(search_query="NORM")
    assert len(results_q) >= 1


def test_delete_analysis(test_db):
    """Test deleting an analysis record."""
    test_db.save_analysis({
        "analysis_id": "MVA-DEL-999",
        "image_hash": "c" * 64,
        "image_filename": "delete_me.png",
        "image_path": "/path/delete_me.png",
        "predicted_category": "PNEUMONIA",
        "confidence_score": 0.95,
        "normal_probability": 0.05,
        "pneumonia_probability": 0.95,
        "model_name": "densenet121",
        "model_version": "1.0.0",
        "inference_time_ms": 35.0,
        "analysis_timestamp": "2026-11-18T10:00:00Z",
    })

    assert test_db.get_analysis_by_id("MVA-DEL-999") is not None
    del_ok = test_db.delete_analysis("MVA-DEL-999", delete_files=False)
    assert del_ok is True
    assert test_db.get_analysis_by_id("MVA-DEL-999") is None


def test_statistics(test_db):
    """Test KPI aggregation."""
    stats = test_db.get_statistics()
    assert "total_analyses" in stats
    assert "category_distribution" in stats
