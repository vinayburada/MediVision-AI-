"""
MediVision AI - Unit Tests for Image Preprocessing Pipeline
Tests image validation, corrupted file detection, normalization, and patient-level splitting.
"""
import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
from pathlib import Path

from preprocessing.image_preprocessing import (
    ImagePreprocessor,
    ImageValidationError,
    default_preprocessor,
)
from preprocessing.augmentation import MedicalAugmentation
from preprocessing.image_loader import DatasetManager
from config.settings import settings


@pytest.fixture
def sample_valid_image_bytes():
    """Create in-memory valid PNG image bytes."""
    img_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img_arr, (150, 150), 80, (200, 200, 200), -1)
    _, buf = cv2.imencode(".png", img_arr)
    return buf.tobytes()


@pytest.fixture
def sample_corrupt_bytes():
    """Corrupted byte sequence simulating broken upload."""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"bad_corrupt_bytes_here"


def test_validate_image_file_valid(sample_valid_image_bytes):
    """Valid image bytes should pass validation and return correct metadata."""
    meta = ImagePreprocessor.validate_image_file(sample_valid_image_bytes)
    assert meta["valid"] is True
    assert meta["original_width"] == 300
    assert meta["original_height"] == 300
    assert meta["channels"] == 3
    assert len(meta["sha256"]) == 64


def test_validate_image_file_corrupt(sample_corrupt_bytes):
    """Corrupted bytes must trigger an ImageValidationError."""
    with pytest.raises(ImageValidationError):
        ImagePreprocessor.validate_image_file(sample_corrupt_bytes)


def test_validate_image_file_empty():
    """Empty or truncated bytes must raise ImageValidationError."""
    with pytest.raises(ImageValidationError):
        ImagePreprocessor.validate_image_file(b"")


def test_load_and_preprocess_tensor_shape(sample_valid_image_bytes):
    """Preprocessing must output exact target dimensions and tensor shape (1, 3, 224, 224)."""
    resized_rgb, float_img, tensor_batch = default_preprocessor.load_and_preprocess(sample_valid_image_bytes)

    assert resized_rgb.shape == (224, 224, 3)
    assert float_img.shape == (224, 224, 3)
    assert float_img.min() >= 0.0 and float_img.max() <= 1.0
    assert tensor_batch.shape == (1, 3, 224, 224)
    assert tensor_batch.dtype == np.float32


def test_denormalize_for_display():
    """Denormalization should return uint8 image in range [0, 255]."""
    dummy_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
    denorm = default_preprocessor.denormalize_for_display(dummy_tensor)
    assert denorm.shape == (224, 224, 3)
    assert denorm.dtype == np.uint8


def test_extract_patient_id():
    """Test patient ID extraction patterns to prevent patient-level leakage."""
    dm = DatasetManager()
    assert dm.extract_patient_id("person1946_bacteria_4874.jpeg", "PNEUMONIA") == "person1946"
    assert dm.extract_patient_id("person20_virus_51.jpeg", "PNEUMONIA") == "person20"
    assert dm.extract_patient_id("NORMAL2-IM-0378-0001.jpeg", "NORMAL") == "normal2-im-0378"


def test_opencv_train_aug():
    """Test fallback OpenCV augmentation returns valid uint8 array without shape change."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    aug = MedicalAugmentation.apply_opencv_train_aug(img)
    assert aug.shape == (224, 224, 3)
    assert aug.dtype == np.uint8
