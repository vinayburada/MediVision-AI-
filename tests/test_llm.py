"""
MediVision AI - Unit Tests for LLM & Report Generator
"""
import pytest
import numpy as np

from llm.gemini_client import GeminiClient
from llm.report_generator import ReportGenerator


def test_gemini_client_fallback_mode():
    """When API key is not configured, client must cleanly return deterministic educational text."""
    client = GeminiClient(api_key="")
    assert client.is_configured is False

    res = client.generate_educational_explanation(
        predicted_category="PNEUMONIA",
        confidence_score=0.9123,
        normal_probability=0.0877,
        pneumonia_probability=0.9123,
        model_name="DenseNet121",
        gradcam_description="Concentrated activations in the right lower lobe.",
    )

    assert res["title"] == "AI-GENERATED EDUCATIONAL EXPLANATION"
    assert "PNEUMONIA" in res["executive_summary"]
    assert "educational_disclaimer" in res
    assert "NOT FOR" in res["educational_disclaimer"].upper() or "NOT A" in res["educational_disclaimer"].upper()


def test_report_generator_hotspots():
    """Test quadrant spatial description for Grad-CAM activations."""
    heatmap = np.zeros((10, 10), dtype=np.float32)
    # Put hotspot in bottom-left quadrant (anatomical right lower zone)
    heatmap[6:10, 0:4] = 0.95

    desc = ReportGenerator.describe_gradcam_hotspots(heatmap)
    assert "anatomical right lower lung zone" in desc


def test_report_generator_full():
    """Verify combined pipeline output schema."""
    gen = ReportGenerator()
    dummy_pred = {
        "predicted_category": "NORMAL",
        "confidence_score": 0.885,
        "class_probabilities": {"NORMAL": 0.885, "PNEUMONIA": 0.115},
        "model_name": "DenseNet121",
    }
    dummy_heatmap = np.full((10, 10), 0.1, dtype=np.float32)

    output = gen.generate_full_explanation(dummy_pred, dummy_heatmap)
    assert "prediction" in output
    assert "educational_explanation" in output
    assert "executive_summary" in output["educational_explanation"]
