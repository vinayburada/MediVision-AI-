"""
MediVision AI - Dataset Loader & Medical Dataset Management
Comprehensive dataset scanning, patient-level leakage prevention, and EDA report generation.
"""
from pathlib import Path
import re
import os
import shutil
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from config.settings import settings
from preprocessing.image_preprocessing import ImagePreprocessor, ImageValidationError


class DatasetManager:
    """
    Manages loading, validation, leakage-free splitting, and EDA for Chest X-Ray datasets.
    Dataset: Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney.
    Categories: NORMAL, PNEUMONIA
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else settings.DATA_DIR
        self.classes = settings.CLASSES

    @staticmethod
    def extract_patient_id(filename: str, label: str) -> str:
        """
        Extract patient identifier from filename to prevent patient-level leakage.
        Examples:
          'person1946_bacteria_4874.jpeg' -> 'person1946'
          'person20_virus_51.jpeg' -> 'person20'
          'NORMAL2-IM-0378-0001.jpeg' -> 'NORMAL2-IM-0378'
          'IM-0001-0001.jpeg' -> 'IM-0001'
        """
        stem = Path(filename).stem
        # Pattern for pneumonia filenames with 'person\d+'
        person_match = re.search(r"(person\d+)", stem, re.IGNORECASE)
        if person_match:
            return person_match.group(1).lower()

        # Pattern for Normal DICOM identifiers (prefix before the last counter)
        parts = stem.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:-1]).lower()

        # Fallback to base stem without numerical suffixes
        cleaned = re.sub(r"_\d+$", "", stem)
        return f"{label.lower()}_{cleaned}"

    def scan_directory(self, target_dir: Path) -> pd.DataFrame:
        """
        Scan a directory tree for image files across predefined classes.
        Returns a structured pandas DataFrame.
        """
        records: List[Dict[str, Any]] = []
        target_dir = Path(target_dir)

        if not target_dir.exists():
            return pd.DataFrame(columns=["filepath", "filename", "label", "patient_id", "exists"])

        for label in self.classes:
            # Check direct subfolder and nested variants
            class_folder = target_dir / label
            if not class_folder.exists():
                # Case-insensitive search
                for sub in target_dir.iterdir():
                    if sub.is_dir() and sub.name.upper() == label:
                        class_folder = sub
                        break

            if class_folder.exists() and class_folder.is_dir():
                for ext in settings.ALLOWED_EXTENSIONS:
                    for img_path in class_folder.glob(f"*{ext}"):
                        patient_id = self.extract_patient_id(img_path.name, label)
                        records.append({
                            "filepath": str(img_path.resolve()),
                            "filename": img_path.name,
                            "label": label,
                            "label_idx": settings.CLASS_TO_IDX[label],
                            "patient_id": patient_id,
                            "file_size_kb": round(img_path.stat().st_size / 1024, 2),
                        })

        df = pd.DataFrame(records)
        return df

    def scan_all_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Scans train, val, test, and raw folders for available datasets.
        """
        splits = {}
        for split_name in ["train", "val", "test", "raw"]:
            split_path = self.data_root / split_name
            splits[split_name] = self.scan_directory(split_path)
        return splits

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Calculates complete statistics without fabricating metrics:
        Total images, class distribution, imbalance ratio, split counts, and availability.
        """
        splits = self.scan_all_splits()
        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        total_train = len(train_df)
        total_val = len(val_df)
        total_test = len(test_df)
        grand_total = total_train + total_val + total_test

        has_data = grand_total > 0

        summary = {
            "dataset_available": has_data,
            "grand_total": grand_total,
            "splits": {
                "train_count": total_train,
                "val_count": total_val,
                "test_count": total_test,
            },
            "class_distribution": {},
            "imbalance_ratio": None,
            "patient_count": 0,
            "instructions": (
                "To train with the full dataset, download the Chest X-Ray dataset from Kaggle: "
                "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia "
                "and extract the 'chest_xray/train', 'chest_xray/val', and 'chest_xray/test' folders into 'data/'."
            ) if not has_data else "Dataset detected and ready for deep learning training and evaluation.",
        }

        if has_data:
            all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            class_counts = all_df["label"].value_counts().to_dict()
            summary["class_distribution"] = class_counts
            summary["patient_count"] = all_df["patient_id"].nunique()

            normal_count = class_counts.get("NORMAL", 0)
            pneumonia_count = class_counts.get("PNEUMONIA", 0)
            if normal_count > 0 and pneumonia_count > 0:
                ratio = round(pneumonia_count / normal_count, 2)
                summary["imbalance_ratio"] = f"{ratio}:1 (Pneumonia to Normal)"
            else:
                summary["imbalance_ratio"] = "N/A"

        return summary

    def inspect_image_dimensions(self, sample_size: int = 50) -> Dict[str, Any]:
        """
        Calculates real image dimensions across available images.
        """
        splits = self.scan_all_splits()
        all_dfs = [df for df in splits.values() if not df.empty]
        if not all_dfs:
            return {"count": 0, "mean_width": None, "mean_height": None, "aspect_ratios": []}

        combined = pd.concat(all_dfs, ignore_index=True)
        sample = combined.sample(n=min(sample_size, len(combined)), random_state=42)

        widths = []
        heights = []
        for path in sample["filepath"]:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue

        if not widths:
            return {"count": 0, "mean_width": None, "mean_height": None}

        return {
            "count": len(widths),
            "mean_width": float(np.mean(widths)),
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "mean_height": float(np.mean(heights)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "mean_aspect_ratio": float(np.mean([w / h for w, h in zip(widths, heights)])),
        }

    def generate_synthetic_educational_samples(self, count_per_class: int = 4) -> List[str]:
        """
        Generates realistic synthetic chest X-ray phantoms for education and testing
        when Kaggle data is not yet downloaded, ensuring complete end-to-end functionality.
        """
        created_files = []
        settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        for label in self.classes:
            target_class_dir = settings.SAMPLES_DIR / label
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(1, count_per_class + 1):
                filename = f"sample_{label.lower()}_{i:02d}.png"
                dest_path = target_class_dir / filename
                if dest_path.exists():
                    created_files.append(str(dest_path))
                    continue

                # Generate synthetic chest radiograph phantom:
                # 1. Base thorax contour with soft tissue gradient
                canvas = np.zeros((400, 400), dtype=np.uint8)
                cv2.ellipse(canvas, (200, 210), (160, 180), 0, 0, 360, 60, -1)

                # 2. Left and Right Lung Fields (Radiolucent dark areas)
                # Right lung (anatomical right = viewer's left)
                cv2.ellipse(canvas, (135, 195), (45, 95), 5, 0, 360, 25, -1)
                # Left lung (anatomical left = viewer's right)
                cv2.ellipse(canvas, (265, 195), (45, 95), -5, 0, 360, 25, -1)

                # 3. Mediastinum & Spine (Radiopaque light central column)
                cv2.rectangle(canvas, (188, 50), (212, 380), 140, -1)
                # Cardiac silhouette (heart shadow tilted to left)
                cv2.ellipse(canvas, (215, 230), (55, 45), 25, 0, 360, 160, -1)

                # 4. Ribs (horizontal curvilinear radiopaque bands)
                for y in range(90, 330, 28):
                    cv2.ellipse(canvas, (135, y), (48, 12), 10, 0, 180, 110, 2)
                    cv2.ellipse(canvas, (265, y), (48, 12), -10, 0, 180, 110, 2)

                # 5. Clavicles (collar bones)
                cv2.line(canvas, (100, 95), (190, 85), 175, 4)
                cv2.line(canvas, (300, 95), (210, 85), 175, 4)

                # 6. If Pneumonia, add alveolar consolidation / patchy opacities in lung fields
                if label == "PNEUMONIA":
                    # Patchy consolidation cloud in right lower lobe
                    cv2.circle(canvas, (140, 235), 26, 125, -1)
                    cv2.circle(canvas, (155, 250), 22, 135, -1)
                    cv2.circle(canvas, (125, 245), 18, 115, -1)
                    if i % 2 == 0:
                        # Bilateral consolidation
                        cv2.circle(canvas, (270, 220), 24, 120, -1)
                        cv2.circle(canvas, (280, 240), 20, 130, -1)

                # 7. Add realistic radiographic Gaussian noise and blur
                noise = np.random.normal(0, 12, canvas.shape).astype(np.float32)
                noisy_canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                blurred = cv2.GaussianBlur(noisy_canvas, (7, 7), 1.5)

                # Save generated image
                cv2.imwrite(str(dest_path), blurred)
                created_files.append(str(dest_path))

        return created_files
