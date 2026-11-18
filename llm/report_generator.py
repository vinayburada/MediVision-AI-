"""
MediVision AI - Report Narrative Generator
Coordinates deep learning outputs, Grad-CAM hotspot spatial localization, and LLM explanation.
"""
from typing import Dict, Any, Optional
import numpy as np

from config.settings import settings
from llm.gemini_client import gemini_client, GeminiClient


class ReportGenerator:
    """Combines vision model metrics, explainability spatial patterns, and LLM summaries."""

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or gemini_client

    @staticmethod
    def describe_gradcam_hotspots(heatmap_2d: np.ndarray) -> str:
        """
        Analyzes 2D Grad-CAM heatmap array to generate an objective spatial description
        of where the model attended without fabricating clinical findings.
        """
        if heatmap_2d is None or heatmap_2d.size == 0:
            return "Diffuse activation across pulmonary fields."

        h, w = heatmap_2d.shape
        # Divide into quadrants: Left/Right (anatomical right = image left) and Upper/Lower
        mid_y, mid_x = h // 2, w // 2

        right_upper = np.mean(heatmap_2d[:mid_y, :mid_x])
        right_lower = np.mean(heatmap_2d[mid_y:, :mid_x])
        left_upper = np.mean(heatmap_2d[:mid_y, mid_x:])
        left_lower = np.mean(heatmap_2d[mid_y:, mid_x:])

        quadrants = {
            "anatomical right upper lung zone (image top-left)": right_upper,
            "anatomical right lower lung zone (image bottom-left)": right_lower,
            "anatomical left upper lung zone (image top-right)": left_upper,
            "anatomical left lower lung zone (image bottom-right)": left_lower,
        }

        # Sort by activation intensity
        sorted_zones = sorted(quadrants.items(), key=lambda x: x[1], reverse=True)
        top_zone, top_val = sorted_zones[0]
        second_zone, second_val = sorted_zones[1]

        if top_val < 0.2:
            return "Low, diffuse activation across the entire thoracic field with no localized peak."

        return (
            f"Primary mathematical activation concentrated in the {top_zone} "
            f"(relative intensity: {top_val:.2f}), with secondary weighting in the {second_zone} "
            f"(relative intensity: {second_val:.2f})."
        )

    def generate_full_explanation(
        self,
        prediction_result: Dict[str, Any],
        heatmap_2d: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates spatial description and LLM response generation.
        """
        predicted_category = prediction_result.get("predicted_category", "UNKNOWN")
        confidence_score = float(prediction_result.get("confidence_score", 0.0))
        class_probs = prediction_result.get("class_probabilities", {})
        normal_prob = float(class_probs.get("NORMAL", 0.5))
        pneumonia_prob = float(class_probs.get("PNEUMONIA", 0.5))
        model_name = prediction_result.get("model_name", "DenseNet121")

        if heatmap_2d is not None:
            spatial_desc = self.describe_gradcam_hotspots(heatmap_2d)
        else:
            spatial_desc = "Diffuse activation across pulmonary regions."

        explanation = self.client.generate_educational_explanation(
            predicted_category=predicted_category,
            confidence_score=confidence_score,
            normal_probability=normal_prob,
            pneumonia_probability=pneumonia_prob,
            model_name=model_name,
            gradcam_description=spatial_desc,
        )

        return {
            "prediction": prediction_result,
            "spatial_hotspot_description": spatial_desc,
            "educational_explanation": explanation,
        }


# Singleton report generator
report_generator = ReportGenerator()
