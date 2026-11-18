"""
MediVision AI - Deep Learning Training Pipeline
Patient-level split, class-imbalance weighting, early stopping, and checkpoint saving.
"""
import argparse
from pathlib import Path
import time
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config.settings import settings
from preprocessing.image_preprocessing import ImagePreprocessor, default_preprocessor
from preprocessing.augmentation import MedicalAugmentation
from preprocessing.image_loader import DatasetManager
from models.architecture import build_model
from models.model_utils import (
    calculate_metrics,
    save_checkpoint,
    get_device,
)


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for Chest Radiographs with patient-level grouping."""

    def __init__(self, dataframe: pd.DataFrame, transform=None, is_train: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        self.preprocessor = default_preprocessor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label_idx"])

        try:
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
        except Exception:
            # Fallback to OpenCV if PIL fails
            import cv2
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                # Return empty tensor if file corrupted
                return torch.zeros((3, *settings.IMAGE_SIZE), dtype=torch.float32), label
            rgb_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            _, _, tensor_np = self.preprocessor.load_and_preprocess(rgb_img)
            tensor = torch.from_numpy(tensor_np).squeeze(0)

        return tensor, label


class Trainer:
    """End-to-end Trainer with patient-level stratification, early stopping, and metric logging."""

    def __init__(
        self,
        model_name: str = settings.DEFAULT_MODEL_NAME,
        batch_size: int = settings.BATCH_SIZE,
        learning_rate: float = settings.LEARNING_RATE,
        num_epochs: int = settings.NUM_EPOCHS,
        patience: int = settings.EARLY_STOPPING_PATIENCE,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device or get_device()

        self.model = build_model(model_name=self.model_name, pretrained=True).to(self.device)
        self.dataset_manager = DatasetManager()

    def prepare_data_loaders(
        self,
        use_sample_if_empty: bool = True,
        smoke_test: bool = False,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Creates train, val, test loaders respecting patient-level isolation.
        """
        splits = self.dataset_manager.scan_all_splits()
        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        # If data directories are empty, generate/use synthetic educational sample X-rays
        if len(train_df) == 0 and use_sample_if_empty:
            self.dataset_manager.generate_synthetic_educational_samples(count_per_class=6)
            samples_df = self.dataset_manager.scan_directory(settings.SAMPLES_DIR)
            if len(samples_df) > 0:
                # Patient-level partition
                patients = np.array(samples_df["patient_id"].unique(), dtype=str)
                np.random.seed(42)
                np.random.shuffle(patients)
                n_train = max(1, int(len(patients) * 0.6))
                train_patients = set(patients[:n_train])
                val_patients = set(patients[n_train:])

                train_df = samples_df[samples_df["patient_id"].isin(train_patients)].copy()
                val_df = samples_df[samples_df["patient_id"].isin(val_patients)].copy()
                test_df = val_df.copy()

        if smoke_test:
            train_df = train_df.head(min(8, len(train_df)))
            val_df = val_df.head(min(4, len(val_df)))
            test_df = test_df.head(min(4, len(test_df)))

        train_transform = MedicalAugmentation.get_train_transforms()
        eval_transform = MedicalAugmentation.get_eval_transforms()

        train_dataset = ChestXRayDataset(train_df, transform=train_transform, is_train=True)
        val_dataset = ChestXRayDataset(val_df, transform=eval_transform, is_train=False)
        test_dataset = ChestXRayDataset(test_df, transform=eval_transform, is_train=False) if len(test_df) > 0 else None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) if test_dataset else None

        return train_loader, val_loader, test_loader

    def compute_class_weights(self, train_loader: DataLoader) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights to counterbalance clinical class imbalance."""
        counts = [0, 0]
        for _, labels in train_loader:
            for lbl in labels:
                counts[int(lbl.item())] += 1

        total = sum(counts)
        if counts[0] > 0 and counts[1] > 0:
            weights = [total / (2.0 * counts[0]), total / (2.0 * counts[1])]
            return torch.tensor(weights, dtype=torch.float32).to(self.device)
        return None

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += int(torch.sum(preds == labels).item())
            total_samples += images.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return float(epoch_loss), float(epoch_acc)

    def evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluates model performance and computes full clinical evaluation metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        y_true = []
        y_pred = []
        y_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_probs.extend(probs[:, 1].cpu().tolist())
                total_samples += images.size(0)

        eval_loss = total_loss / max(1, total_samples)
        metrics = calculate_metrics(y_true, y_pred, y_probs)
        return float(eval_loss), metrics

    def run_training(
        self,
        use_sample_if_empty: bool = True,
        smoke_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes full training loop with early stopping and checkpoint saving.
        """
        train_loader, val_loader, test_loader = self.prepare_data_loaders(
            use_sample_if_empty=use_sample_if_empty,
            smoke_test=smoke_test,
        )

        class_weights = self.compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=settings.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        patience_counter = 0

        checkpoint_filename = (
            settings.BEST_MODEL_FILENAME if "densenet" in self.model_name else settings.VIT_MODEL_FILENAME
        )
        checkpoint_path = settings.CHECKPOINTS_DIR / checkpoint_filename

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_metrics = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_score"]

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_f1": round(val_f1, 4),
            }
            history.append(record)

            # Checkpoint on best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    filepath=checkpoint_path,
                    model_name=self.model_name,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        history_df = pd.DataFrame(history)
        history_csv_path = settings.PLOTS_DIR / f"{self.model_name}_training_history.csv"
        history_df.to_csv(history_csv_path, index=False)

        # Final evaluation on test loader if present
        test_metrics = None
        if test_loader is not None:
            _, test_metrics = self.evaluate(test_loader, criterion)

        return {
            "model_name": self.model_name,
            "epochs_completed": len(history),
            "best_val_loss": round(best_val_loss, 4),
            "checkpoint_path": str(checkpoint_path),
            "history_csv": str(history_csv_path),
            "history": history,
            "test_metrics": test_metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="MediVision AI - Deep Learning Model Training")
    parser.add_argument("--model", type=str, default=settings.DEFAULT_MODEL_NAME, help="Model architecture")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification run on small subset")
    args = parser.parse_args()

    trainer = Trainer(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )
    result = trainer.run_training(smoke_test=args.smoke_test)
    print(f"Training completed successfully for {result['model_name']}. Checkpoint saved at: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
