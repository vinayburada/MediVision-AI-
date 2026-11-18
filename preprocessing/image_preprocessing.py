"""
MediVision AI - Image Preprocessing Module
Robust medical image loading, validation, OpenCV transformation, and tensor conversion.
"""
from pathlib import Path
import hashlib
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from config.settings import settings


class ImageValidationError(Exception):
    """Raised when an uploaded image fails integrity or format checks."""
    pass


class ImagePreprocessor:
    """Handles medical image loading, sanitization, OpenCV transformation, and normalization."""

    def __init__(
        self,
        target_size: Tuple[int, int] = settings.IMAGE_SIZE,
        mean: list = settings.IMAGENET_MEAN,
        std: list = settings.IMAGENET_STD,
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    @staticmethod
    def compute_sha256(image_bytes: bytes) -> str:
        """Compute SHA-256 hash of image bytes for deduplication and audit tracking."""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def validate_image_file(file_path_or_bytes: Union[str, Path, bytes], max_mb: int = settings.MAX_UPLOAD_SIZE_MB) -> Dict[str, Any]:
        """
        Validate image integrity, header, format, and dimension sanity.
        Raises ImageValidationError if invalid or corrupted.
        """
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                raise ImageValidationError(f"Image file does not exist: {path}")
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                raise ImageValidationError(f"File size ({file_size_mb:.2f} MB) exceeds maximum limit ({max_mb} MB).")
            if path.suffix.lower() not in settings.ALLOWED_EXTENSIONS:
                raise ImageValidationError(f"Unsupported file extension '{path.suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}")
            with open(path, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = file_path_or_bytes
            file_size_mb = len(raw_bytes) / (1024 * 1024)
            if file_size_mb > max_mb:
                raise ImageValidationError(f"File size ({file_size_mb:.2f} MB) exceeds maximum limit ({max_mb} MB).")

        if len(raw_bytes) < 64:
            raise ImageValidationError("File is truncated or empty.")

        # Test PIL opening & verification
        import io
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_format = pil_img.format
                width, height = pil_img.size
                mode = pil_img.mode
                pil_img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ImageValidationError(f"Corrupted or invalid image data: {str(exc)}") from exc

        # Test OpenCV decoding
        np_buf = np.frombuffer(raw_bytes, np.uint8)
        cv_img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
        if cv_img is None or cv_img.size == 0:
            raise ImageValidationError("OpenCV failed to decode image. Corrupted raster data.")

        if width < 32 or height < 32:
            raise ImageValidationError(f"Image dimensions ({width}x{height}) are too small for clinical analysis.")

        return {
            "valid": True,
            "format": pil_format,
            "original_width": width,
            "original_height": height,
            "aspect_ratio": round(width / height, 3),
            "channels": 1 if len(cv_img.shape) == 2 else cv_img.shape[2],
            "file_size_kb": round(len(raw_bytes) / 1024, 2),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }

    def load_and_preprocess(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads an image from various input sources, performs OpenCV standardization,
        and returns:
            1. original_bgr (np.ndarray): Original image resized to target for visualization
            2. normalized_rgb (np.ndarray): Normalized float32 image (H, W, 3) in [0, 1] range
            3. tensor_input (np.ndarray or torch.Tensor): shape (1, 3, 224, 224) ready for model
        """
        # Step 1: Decode/Convert to standard 3-channel RGB image
        if isinstance(source, (str, Path)):
            path_str = str(source)
            cv_img = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ImageValidationError(f"Failed to read image from path: {path_str}")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        elif isinstance(source, bytes):
            np_buf = np.frombuffer(source, np.uint8)
            cv_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ImageValidationError("Failed to decode image bytes with OpenCV.")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        elif isinstance(source, Image.Image):
            rgb_img = np.array(source.convert("RGB"))

        elif isinstance(source, np.ndarray):
            if len(source.shape) == 2:
                # Grayscale to RGB
                rgb_img = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            elif len(source.shape) == 3 and source.shape[2] == 4:
                # RGBA to RGB
                rgb_img = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
            elif len(source.shape) == 3 and source.shape[2] == 3:
                rgb_img = source.copy()
            else:
                raise ImageValidationError(f"Unsupported array shape for image: {source.shape}")
        else:
            raise TypeError(f"Unsupported input type for source: {type(source)}")

        # Step 2: Resize with OpenCV high-quality bicubic interpolation
        resized_rgb = cv2.resize(rgb_img, self.target_size, interpolation=cv2.INTER_CUBIC)

        # Step 3: Convert to float in range [0, 1]
        float_img = resized_rgb.astype(np.float32) / 255.0

        # Step 4: Apply ImageNet Standardization (x - mean) / std
        standardized = (float_img - self.mean) / self.std

        # Step 5: Permute dimensions from (H, W, C) to (C, H, W) and add batch dimension (1, C, H, W)
        tensor_array = np.transpose(standardized, (2, 0, 1))
        tensor_batch = np.expand_dims(tensor_array, axis=0).astype(np.float32)

        return resized_rgb, float_img, tensor_batch

    def denormalize_for_display(self, tensor_or_array: np.ndarray) -> np.ndarray:
        """
        Reverse standardization back to [0, 255] uint8 RGB image for visualization.
        """
        if tensor_or_array.ndim == 4:
            arr = tensor_or_array[0]
        else:
            arr = tensor_or_array

        if arr.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            arr = np.transpose(arr, (1, 2, 0))

        denorm = (arr * self.std) + self.mean
        denorm = np.clip(denorm * 255.0, 0, 255).astype(np.uint8)
        return denorm


# Singleton default preprocessor instance
default_preprocessor = ImagePreprocessor()
