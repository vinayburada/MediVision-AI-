"""
MediVision AI - Gemini Generative AI Client
Structured prompt engineering for educational medical AI explanations.
Strictly adheres to non-diagnostic educational guidelines.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

from config.settings import settings

logger = logging.getLogger("medivision.llm")


SYSTEM_PROMPT = """
You are the MediVision AI Educational Science Communicator, an AI assistant dedicated exclusively to explaining computer vision research concepts to medical students, researchers, and engineers.

CRITICAL SAFETY & ETHICAL RULES:
1. You are an EDUCATIONAL EXPLAINER, NOT a physician or medical diagnostic device.
2. NEVER formulate a medical diagnosis, clinical opinion, or triage recommendation.
3. NEVER prescribe, suggest, or comment on clinical treatments, medications, or surgical interventions.
4. DO NOT invent or fabricate clinical symptoms or radiological observations not provided in the model metrics.
5. Emphasize that the provided classification is a statistical pattern matched by a deep neural network (DenseNet121 / ViT) trained on a specific academic dataset.
6. Clearly explain that Grad-CAM heatmaps highlight pixels influencing the mathematical classification function, NOT confirmed pathological boundaries.
7. Output MUST strictly adhere to the requested JSON schema.
"""


class GeminiClient:
    """Interface to Google Gemini API with fallback for offline/unconfigured environments."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = settings.GEMINI_MODEL):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self.is_configured = bool(self.api_key and self.api_key.strip() and self.api_key != "your_gemini_api_key_here")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client if key is configured."""
        if not self.is_configured:
            return

        try:
            # Try new google.genai client first
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._client_type = "genai"
        except Exception:
            try:
                # Fallback to google.generativeai
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._client = gai.GenerativeModel(self.model_name)
                self._client_type = "generativeai"
            except Exception as e:
                logger.warning(f"Could not initialize Gemini API client: {e}")
                self.is_configured = False

    def generate_educational_explanation(
        self,
        predicted_category: str,
        confidence_score: float,
        normal_probability: float,
        pneumonia_probability: float,
        model_name: str = "DenseNet121",
        gradcam_description: str = "Spatial activation maps over the middle and lower pulmonary lobes.",
    ) -> Dict[str, Any]:
        """
        Generates structured educational explanation using Gemini LLM,
        or deterministic educational scientific template if offline/unconfigured.
        """
        # Fallback template if Gemini API key not provided or unavailable
        if not self.is_configured or self._client is None:
            return self._build_deterministic_explanation(
                predicted_category=predicted_category,
                confidence_score=confidence_score,
                normal_probability=normal_probability,
                pneumonia_probability=pneumonia_probability,
                model_name=model_name,
                gradcam_description=gradcam_description,
                is_fallback=True,
            )

        prompt = f"""
Analyze the following deep learning classification outputs for a chest radiograph and return a valid JSON object:

INPUT PARAMETERS:
- Predicted Category: {predicted_category}
- Confidence Score: {confidence_score:.4f} ({confidence_score * 100:.1f}%)
- Normal Probability: {normal_probability:.4f}
- Pneumonia Probability: {pneumonia_probability:.4f}
- Model Architecture: {model_name}
- Grad-CAM Region Description: {gradcam_description}

JSON SCHEMA REQUIRED:
{{
    "title": "AI-GENERATED EDUCATIONAL EXPLANATION",
    "executive_summary": "High-level summary of the model output and confidence interpretation.",
    "technical_findings": "Detailed technical analysis of the probability distribution and model mechanism.",
    "gradcam_interpretation": "Scientific explanation of why the model attended to highlighted regions.",
    "model_limitations": "Specific limitations of {model_name} on chest radiography (e.g., confounders, artifacts).",
    "dataset_limitations": "Limitations of Mooney Chest X-Ray dataset (pediatric bias, binary classification).",
    "educational_disclaimer": "Prominent statement that this is research only, not a clinical diagnosis."
}}
"""
        try:
            if self._client_type == "genai":
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                    config={"response_mime_type": "application/json"}
                )
                raw_text = response.text
            else:
                response = self._client.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{prompt}",
                    generation_config={"response_mime_type": "application/json"}
                )
                raw_text = response.text

            # Parse JSON
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "", 1).rsplit("```", 1)[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.replace("```", "", 1).rsplit("```", 1)[0].strip()

            parsed = json.loads(cleaned)
            parsed["source"] = "Google Gemini API (Live)"
            return parsed

        except Exception as exc:
            logger.error(f"Gemini API request failed: {exc}. Falling back to deterministic template.")
            return self._build_deterministic_explanation(
                predicted_category=predicted_category,
                confidence_score=confidence_score,
                normal_probability=normal_probability,
                pneumonia_probability=pneumonia_probability,
                model_name=model_name,
                gradcam_description=gradcam_description,
                is_fallback=True,
                error_msg=str(exc),
            )

    def _build_deterministic_explanation(
        self,
        predicted_category: str,
        confidence_score: float,
        normal_probability: float,
        pneumonia_probability: float,
        model_name: str,
        gradcam_description: str,
        is_fallback: bool = True,
        error_msg: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Provides verified educational text when Gemini API key is missing or offline."""
        pct = f"{confidence_score * 100:.1f}%"

        if predicted_category == "PNEUMONIA":
            summary = (
                f"The {model_name} neural network classified this radiograph under the 'PNEUMONIA' category "
                f"with a statistical confidence of {pct}. The network identified pixel density variations "
                "typical of radiographic consolidations or interstitial opacities in the training corpus."
            )
            technical = (
                f"Binary softmax output yielded {pneumonia_probability:.4f} for Pneumonia versus {normal_probability:.4f} "
                "for Normal. The decision boundary was driven by high-frequency features in the penultimate dense block."
            )
        else:
            summary = (
                f"The {model_name} neural network classified this radiograph under the 'NORMAL' category "
                f"with a statistical confidence of {pct}. The features extracted correspond to clear lung fields "
                "and typical radiographic parenchymal patterns present in the baseline dataset."
            )
            technical = (
                f"Binary softmax output yielded {normal_probability:.4f} for Normal versus {pneumonia_probability:.4f} "
                "for Pneumonia. The activation levels in the consolidation-detection filters remained below activation threshold."
            )

        source_label = "Educational Research Engine (Offline Fallback)"
        if error_msg:
            source_label += f" - API Notice: {error_msg[:60]}..."
        elif not self.is_configured:
            source_label += " - (No Gemini API Key Configured in .env)"

        return {
            "title": "AI-GENERATED EDUCATIONAL EXPLANATION",
            "source": source_label,
            "executive_summary": summary,
            "technical_findings": technical,
            "gradcam_interpretation": (
                f"Grad-CAM spatial activation maps revealed that the model's final convolutional layer focused on: "
                f"{gradcam_description} These highlighted coordinates provided the mathematical weight for the classification."
            ),
            "model_limitations": (
                f"{model_name} classifies based on learned spatial feature representations. It does not distinguish "
                "between bacterial and viral etiologies, cannot account for patient medical history or clinical presentation, "
                "and may be susceptible to confounding radiological factors such as patient rotation or thoracic support apparatus."
            ),
            "dataset_limitations": (
                "The Mooney Chest X-Ray dataset contains primarily pediatric anterior-posterior radiographs from a single medical center. "
                "This demographic and acquisition bias limits generalizability to adult populations or diverse clinical imaging protocols."
            ),
            "educational_disclaimer": settings.DISCLAIMER_TEXT,
        }


# Singleton Gemini client instance
gemini_client = GeminiClient()
