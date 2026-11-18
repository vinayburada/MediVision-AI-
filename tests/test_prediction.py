"""
MediVision AI - Unit Tests for Deep Learning Model & Prediction Engine
"""
import pytest
import numpy as np
import torch
import cv2

from models.architecture import build_model, DenseNet121Classifier, VisionTransformerClassifier
from models.predict import Predictor
from models.model_utils import calculate_metrics, get_device
from config.settings import settings


@pytest.fixture
def dummy_image_bytes():
    """Create a 224x224 RGB image in memory."""
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    cv2.circle(img, (112, 112), 40, (200, 200, 200), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_densenet_architecture_forward():
    """Verify DenseNet121 classifier outputs shape (B, 2) on batch input."""
    model = DenseNet121Classifier(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 2)


def test_vit_architecture_forward():
    """Verify Vision Transformer classifier outputs shape (B, 2)."""
    model = VisionTransformerClassifier(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 2)


def test_predictor_inference(dummy_image_bytes):
    """Test full Predictor inference cycle."""
    predictor = Predictor(model_name="densenet121")
    result = predictor.predict(dummy_image_bytes)

    assert result["predicted_category"] in settings.CLASSES
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert "NORMAL" in result["class_probabilities"]
    assert "PNEUMONIA" in result["class_probabilities"]

    # Softmax probabilities should sum to 1.0 (approx)
    prob_sum = result["class_probabilities"]["NORMAL"] + result["class_probabilities"]["PNEUMONIA"]
    assert pytest.approx(prob_sum, abs=1e-3) == 1.0
    assert result["inference_time_ms"] > 0.0


def test_calculate_metrics():
    """Verify metrics calculation without fabrication."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    y_probs = [0.1, 0.2, 0.8, 0.9, 0.4]

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    assert metrics["accuracy"] == 0.8
    assert metrics["confusion_matrix"] == [[2, 0], [1, 2]]
    assert metrics["f1_score"] > 0.0
    assert metrics["sample_count"] == 5
