"""
MediVision AI - Deep Learning Inference Module
Real-time image classification, confidence estimation, and latency profiling.
"""
from pathlib import Path
import time
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.settings import settings
from preprocessing.image_preprocessing import ImagePreprocessor, default_preprocessor, ImageValidationError
from models.model_utils import load_checkpoint, get_device
from models.architecture import build_model


class Predictor:
    """
    Production inference engine for Chest X-Ray classification.
    Supports DenseNet121 and Vision Transformer architectures.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: str = settings.DEFAULT_MODEL_NAME,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model_name = model_name
        self.preprocessor = default_preprocessor
        self.classes = settings.CLASSES
        self.model = None
        self.metadata = {}
        self.is_loaded = False

        # Attempt to load checkpoint if specified or available
        target_ckpt = checkpoint_path
        if target_ckpt is None:
            default_path = settings.CHECKPOINTS_DIR / settings.BEST_MODEL_FILENAME
            if default_path.exists():
                target_ckpt = default_path

        if target_ckpt and Path(target_ckpt).exists():
            try:
                self.model, self.metadata = load_checkpoint(target_ckpt, device=self.device)
                self.model_name = self.metadata.get("model_name", model_name)
                self.is_loaded = True
            except Exception as e:
                # If checkpoint loading encounters issue, instantiate backbone
                self._fallback_init()
        else:
            self._fallback_init()

    def _fallback_init(self):
        """Instantiate architecture when checkpoint is not yet saved."""
        try:
            self.model = build_model(self.model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.metadata = {
                "model_name": self.model_name,
                "version": f"{settings.VERSION}-untrained-backbone",
                "status": "Pretrained weights loaded without fine-tuning checkpoint",
            }
        except Exception:
            self.is_loaded = False

    def predict(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on an input image.
        Returns prediction category, probabilities, confidence, and inference time.
        """
        if self.model is None or not self.is_loaded:
            raise RuntimeError(
                "No deep learning model is initialized. Please ensure weights or checkpoints are available."
            )

        # 1. Validation and Preprocessing
        original_rgb, float_rgb, tensor_np = self.preprocessor.load_and_preprocess(image_source)

        # 2. Conversion to torch tensor on target hardware
        input_tensor = torch.from_numpy(tensor_np).to(self.device)

        # 3. Model Inference with execution timing
        start_time = time.perf_counter()
        self.model.eval()

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Extract class-level scores
        normal_prob = float(probabilities[0])
        pneumonia_prob = float(probabilities[1])

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # 5. Calculate prediction margin / uncertainty
        margin = abs(pneumonia_prob - normal_prob)

        return {
            "predicted_category": predicted_class,
            "predicted_index": predicted_idx,
            "confidence_score": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "class_probabilities": {
                "NORMAL": round(normal_prob, 4),
                "PNEUMONIA": round(pneumonia_prob, 4),
            },
            "margin": round(margin, 4),
            "inference_time_ms": round(elapsed_ms, 2),
            "model_name": self.model_name,
            "device": str(self.device),
            "input_tensor": input_tensor,
            "original_rgb": original_rgb,
        }
