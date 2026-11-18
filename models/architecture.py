"""
MediVision AI - Deep Learning Model Architectures
Includes transfer learning implementations of DenseNet121 and Vision Transformer (ViT-B/16).
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from config.settings import settings


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 Architecture adapted for Chest X-Ray Binary Classification (NORMAL vs PNEUMONIA).
    Utilizes dense connectivity with feature reuse across dense blocks.
    Target layer for Grad-CAM: features.denseblock4.denselayer16.conv2
    """

    def __init__(
        self,
        num_classes: int = settings.NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = "densenet121"
        self.num_classes = num_classes

        # Load DenseNet121
        if pretrained:
            try:
                weights = models.DenseNet121_Weights.DEFAULT
                self.backbone = models.densenet121(weights=weights)
            except Exception:
                # Offline or weight download issue fallback
                self.backbone = models.densenet121(weights=None)
        else:
            self.backbone = models.densenet121(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # In DenseNet121, in_features is 1024
        in_features = self.backbone.classifier.in_features

        # Replace classification head with modern medical MLP
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    @property
    def features(self):
        """Access convolutional feature extraction blocks."""
        return self.backbone.features

    def get_gradcam_target_layer(self):
        """Returns the penultimate high-resolution convolutional layer for Grad-CAM."""
        return self.backbone.features.denseblock4.denselayer16.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits."""
        return self.backbone(x)


class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for Comparative Analysis.
    Applies multi-head self-attention mechanisms over 16x16 image patches.
    """

    def __init__(
        self,
        num_classes: int = settings.NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = "vit_b_16"
        self.num_classes = num_classes

        if pretrained:
            try:
                weights = models.ViT_B_16_Weights.DEFAULT
                self.backbone = models.vit_b_16(weights=weights)
            except Exception:
                self.backbone = models.vit_b_16(weights=None)
        else:
            self.backbone = models.vit_b_16(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    model_name: str = "densenet121",
    pretrained: bool = True,
    num_classes: int = settings.NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function to instantiate deep learning models by name.
    """
    name = model_name.lower().replace("-", "").replace("_", "")
    if "densenet" in name:
        return DenseNet121Classifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif "vit" in name or "transformer" in name:
        return VisionTransformerClassifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'. Choose 'densenet121' or 'vit_b_16'.")
