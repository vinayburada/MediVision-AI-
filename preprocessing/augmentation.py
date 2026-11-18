"""
MediVision AI - Data Augmentation Pipeline
Clinically sensible augmentations for medical imaging (preventing inverted anatomies).
"""
import random
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import settings


class MedicalAugmentation:
    """
    Clinically appropriate data augmentations for chest radiography.
    Chest X-rays have fixed anatomical orientation (cranial to caudal).
    Vertical flips are strictly prohibited.
    """

    @staticmethod
    def get_train_transforms():
        """Returns PyTorch torchvision transform composition for training."""
        if not TORCH_AVAILABLE:
            return None

        return transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            # Gentle horizontal flip (mirroring heart/lungs with low probability for data diversity)
            transforms.RandomHorizontalFlip(p=0.5),
            # Small rotation to simulate patient positioning variability in portable X-rays
            transforms.RandomRotation(degrees=(-7, 7)),
            # Mild brightness and contrast shifts to simulate varying exposure times
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
        ])

    @staticmethod
    def get_eval_transforms():
        """Returns deterministic transforms for validation and test sets."""
        if not TORCH_AVAILABLE:
            return None

        return transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
        ])

    @staticmethod
    def apply_opencv_train_aug(image_rgb: np.ndarray) -> np.ndarray:
        """
        Pure OpenCV fallback augmentation when running without PyTorch torchvision.
        Preserves uint8 [0, 255] RGB shape.
        """
        img = image_rgb.copy()
        # Random horizontal flip (p=0.5)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random small rotation between -7 and 7 degrees
        angle = random.uniform(-7, 7)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random brightness and contrast adjustment
        alpha = random.uniform(0.9, 1.1)  # contrast
        beta = random.uniform(-10, 10)     # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img
