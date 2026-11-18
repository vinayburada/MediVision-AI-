"""
MediVision AI - Explainable AI (XAI) with Grad-CAM
Generates class activation heatmaps and overlays for DenseNet121 chest radiography.
"""
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn

from config.settings import settings


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Extracts convolutional feature activations and gradients to identify regions
    influencing deep learning predictions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_gradcam_target_layer"):
                self.target_layer = model.get_gradcam_target_layer()
            elif hasattr(model, "features"):
                # DenseNet default
                self.target_layer = model.features.denseblock4.denselayer16.conv2
            else:
                raise ValueError("A valid convolutional target_layer must be specified for Grad-CAM.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple containing the gradients with respect to the output of the module
            self.gradients = grad_out[0].detach()

        f_hook = self.target_layer.register_forward_hook(forward_hook)
        b_hook = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([f_hook, b_hook])

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a given input tensor and target class.
        Returns:
            heatmap: 2D numpy array normalized in [0, 1] matching feature map dimensions.
        """
        self.model.zero_grad()

        # Forward pass
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class_idx is None:
            # Default to the winning predicted class
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        # Target class score
        score = logits[:, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Gradients shape: (B, C, H, W)
        # Global Average Pooling of gradients across spatial dimensions (H, W) -> weights (B, C, 1, 1)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)

        # Apply ReLU to focus only on positive contributions to the target class
        cam = torch.clamp(cam, min=0.0)

        # Squeeze dimensions to 2D
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] range safely
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        if max_val - min_val > 1e-8:
            heatmap = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap

    @staticmethod
    def overlay_heatmap(
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = settings.HEATMAP_ALPHA,
        colormap: str = settings.HEATMAP_COLORMAP,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize heatmap to match the original image dimensions and blend.
        Args:
            original_image_rgb: uint8 RGB numpy array (H, W, 3)
            heatmap: float 2D numpy array [0, 1]
            alpha: blending weight for heatmap
            colormap: 'JET', 'VIRIDIS', 'TURBO', 'MAGMA', 'PLASMA'
        Returns:
            Tuple of (colored_heatmap_rgb, blended_overlay_rgb)
        """
        h, w = original_image_rgb.shape[:2]

        # Resize heatmap with bilinear interpolation
        resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Colormap selection
        cmap_code = cv2.COLORMAP_JET
        cm_upper = colormap.upper()
        if cm_upper == "VIRIDIS":
            cmap_code = cv2.COLORMAP_VIRIDIS
        elif cm_upper == "TURBO":
            cmap_code = cv2.COLORMAP_TURBO
        elif cm_upper == "MAGMA":
            cmap_code = cv2.COLORMAP_MAGMA
        elif cm_upper == "PLASMA":
            cmap_code = cv2.COLORMAP_PLASMA

        # Apply OpenCV colormap (outputs BGR)
        colored_bgr = cv2.applyColorMap(heatmap_uint8, cmap_code)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image: overlay = alpha * heatmap + (1 - alpha) * original
        original_float = original_image_rgb.astype(np.float32)
        colored_float = colored_rgb.astype(np.float32)
        blended = cv2.addWeighted(colored_float, alpha, original_float, 1.0 - alpha, 0.0)
        overlay_rgb = np.clip(blended, 0, 255).astype(np.uint8)

        return colored_rgb, overlay_rgb

    @staticmethod
    def explain_gradcam_educational() -> Dict[str, str]:
        """Returns standard educational metadata and clinical disclaimers for Grad-CAM."""
        return {
            "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
            "purpose": (
                "Grad-CAM visualizes the spatial attention of the convolutional neural network by "
                "calculating the gradients of the target category score with respect to feature maps in the final dense block."
            ),
            "interpretation_guide": (
                "Warmer regions (red, orange, yellow in JET colormap) indicate pixel groupings that strongly "
                "influenced the penultimate classification layer toward the predicted class. "
                "Cooler regions (blue, purple) contributed minimally to this specific classification decision."
            ),
            "scientific_limitation": (
                "Grad-CAM indicates mathematical correlation, NOT definitive pathological etiology. "
                "High activation may occasionally align with anatomical markers, medical devices (e.g. pacemakers, wires), "
                "or radiological artifacts rather than true focal consolidations. It must not be treated as a clinical boundary or diagnostic finding."
            ),
        }
