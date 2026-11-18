"""
MediVision AI - FastAPI Pydantic Request & Response Schemas
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str
    torch_version: str
    cuda_available: bool
    device: str
    database_connected: bool
    gemini_configured: bool


class ImageValidationResponse(BaseModel):
    valid: bool
    filename: str
    format: str
    original_width: int
    original_height: int
    aspect_ratio: float
    channels: int
    file_size_kb: float
    sha256: str
    temp_path: str


class PredictionResponse(BaseModel):
    predicted_category: str
    predicted_index: int
    confidence_score: float
    confidence_percentage: str
    class_probabilities: Dict[str, float]
    margin: float
    inference_time_ms: float
    model_name: str
    device: str


class GradCAMResponse(BaseModel):
    analysis_id: str
    target_layer: str
    colormap: str
    gradcam_image_base64: str
    gradcam_image_path: str
    method_explanation: Dict[str, str]


class ExplanationRequest(BaseModel):
    predicted_category: str
    confidence_score: float
    normal_probability: float
    pneumonia_probability: float
    model_name: str = "DenseNet121"
    spatial_description: Optional[str] = None


class ExplanationResponse(BaseModel):
    title: str
    source: str
    executive_summary: str
    technical_findings: str
    gradcam_interpretation: str
    model_limitations: str
    dataset_limitations: str
    educational_disclaimer: str


class FullAnalysisResponse(BaseModel):
    analysis_id: str
    image_filename: str
    image_hash: str
    predicted_category: str
    confidence_score: float
    normal_probability: float
    pneumonia_probability: float
    inference_time_ms: float
    model_name: str
    model_version: str
    analysis_timestamp: str
    gradcam_image_path: Optional[str] = None
    pdf_report_path: Optional[str] = None
    educational_explanation: Dict[str, Any]
    disclaimer: str


class AnalysisRecord(BaseModel):
    id: int
    analysis_id: str
    image_hash: str
    image_filename: str
    predicted_category: str
    confidence_score: float
    normal_probability: float
    pneumonia_probability: float
    model_name: str
    model_version: str
    inference_time_ms: float
    analysis_timestamp: str
    gradcam_image_path: Optional[str] = None
    pdf_report_path: Optional[str] = None


class HistoryListResponse(BaseModel):
    total_count: int
    records: List[Dict[str, Any]]


class DatasetSummaryResponse(BaseModel):
    dataset_available: bool
    grand_total: int
    splits: Dict[str, int]
    class_distribution: Dict[str, int]
    imbalance_ratio: Optional[str]
    instructions: str
