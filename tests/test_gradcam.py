"""
MediVision AI - Unit Tests for Grad-CAM Explainable AI Engine
"""
import pytest
import numpy as np
import torch
import cv2

from models.architecture import DenseNet121Classifier
from explainability.gradcam import GradCAM


@pytest.fixture
def dummy_model():
    model = DenseNet121Classifier(num_classes=2, pretrained=False)
    model.eval()
    return model


def test_gradcam_hook_and_generation(dummy_model):
    """Test hook registration, gradient extraction, and heatmap generation."""
    cam = GradCAM(dummy_model)
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    try:
        heatmap_2d = cam.generate_heatmap(input_tensor, target_class_idx=1)
        assert heatmap_2d.ndim == 2
        assert 0.0 <= heatmap_2d.min()
        assert heatmap_2d.max() <= 1.0
    finally:
        cam.remove_hooks()
        assert len(cam._hooks) == 0


def test_gradcam_overlay():
    """Test heatmap color mapping and image alpha blending."""
    orig_img = np.full((224, 224, 3), 100, dtype=np.uint8)
    heatmap = np.linspace(0, 1, 224 * 224).reshape((224, 224)).astype(np.float32)

    colored_rgb, overlay_rgb = GradCAM.overlay_heatmap(orig_img, heatmap, alpha=0.45, colormap="JET")

    assert colored_rgb.shape == (224, 224, 3)
    assert colored_rgb.dtype == np.uint8
    assert overlay_rgb.shape == (224, 224, 3)
    assert overlay_rgb.dtype == np.uint8


def test_explain_gradcam_educational():
    """Verify standard educational metadata contents."""
    info = GradCAM.explain_gradcam_educational()
    assert "method" in info
    assert "purpose" in info
    assert "scientific_limitation" in info
