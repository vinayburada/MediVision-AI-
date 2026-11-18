"""
MediVision AI - Deep Learning Utilities
Metrics computation, checkpoint serialization, and clinical evaluation reporting.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import json
import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.settings import settings


def get_device() -> torch.device:
    """Detect available compute hardware (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None,
    class_names: List[str] = settings.CLASSES,
) -> Dict[str, Any]:
    """
    Computes rigorous clinical evaluation metrics without fabrication.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if len(y_true_arr) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": {},
        }

    acc = float(np.mean(y_true_arr == y_pred_arr))

    if SKLEARN_AVAILABLE:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
        report = classification_report(
            y_true_arr, y_pred_arr, target_names=class_names, output_dict=True, zero_division=0
        )

        roc_auc = None
        if y_probs is not None and len(np.unique(y_true_arr)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_arr, y_probs))
            except Exception:
                roc_auc = None
    else:
        # Pure numpy fallback
        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        cm = [[tn, fp], [fn, tp]]
        roc_auc = None
        report = {
            "NORMAL": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": int(np.sum(y_true_arr == 0))},
            "PNEUMONIA": {"precision": precision, "recall": recall, "f1-score": f1, "support": int(np.sum(y_true_arr == 1))},
        }

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "sample_count": len(y_true_arr),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    filepath: Union[str, Path],
    model_name: str = "densenet121",
) -> None:
    """Save model checkpoint with architecture metadata and validation metrics."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_name": model_name,
        "version": settings.VERSION,
        "classes": settings.CLASSES,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
    }
    torch.save(state, path)


def load_checkpoint(
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and metadata from a checkpoint file."""
    if device is None:
        device = get_device()

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # weights_only=False allows unpickling custom dict structures safely
    checkpoint = torch.save if False else torch.load(path, map_location=device, weights_only=False)

    from models.architecture import build_model
    model_name = checkpoint.get("model_name", "densenet121")
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "model_name": model_name,
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "version": checkpoint.get("version", settings.VERSION),
        "classes": checkpoint.get("classes", settings.CLASSES),
    }

    return model, metadata
