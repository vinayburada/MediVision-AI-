"""
MediVision AI - FastAPI Application Server
Modular RESTful API endpoints for medical image analysis, explainability, and report generation.
All endpoints use synchronous execution for optimal CPU-bound deep learning inference.
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch

from config.settings import settings
from preprocessing.image_preprocessing import (
    ImagePreprocessor,
    default_preprocessor,
    ImageValidationError,
)
from preprocessing.image_loader import DatasetManager
from models.predict import Predictor
from explainability.gradcam import GradCAM
from llm.gemini_client import gemini_client
from llm.report_generator import report_generator
from database.db import db
from utils.pdf_generator import pdf_generator
from utils.helpers import generate_analysis_id, sanitize_filename, image_to_base64
from api.schemas import (
    HealthResponse,
    ImageValidationResponse,
    PredictionResponse,
    GradCAMResponse,
    ExplanationRequest,
    ExplanationResponse,
    FullAnalysisResponse,
    HistoryListResponse,
    DatasetSummaryResponse,
)

app = FastAPI(
    title="MediVision AI REST API",
    description="Intelligent Chest X-Ray Analysis & Explainable AI Education Platform",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model predictor instance
predictor = Predictor()
dataset_manager = DatasetManager()


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns application runtime health, model device, and database connectivity."""
    db_ok = False
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            db_ok = cur.fetchone()[0] == 1
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device=str(predictor.device),
        database_connected=db_ok,
        gemini_configured=gemini_client.is_configured,
    )


@app.post("/api/v1/upload", response_model=ImageValidationResponse, tags=["Image Processing"])
def upload_and_validate(file: UploadFile = File(...)):
    """Validate image integrity, format, dimensions, and save to upload cache."""
    try:
        contents = file.file.read()
        metadata = ImagePreprocessor.validate_image_file(contents)

        safe_name = sanitize_filename(file.filename or "upload.png")
        saved_path = settings.UPLOADS_DIR / f"{metadata['sha256'][:12]}_{safe_name}"
        with open(saved_path, "wb") as f:
            f.write(contents)

        return ImageValidationResponse(
            valid=True,
            filename=safe_name,
            format=metadata["format"],
            original_width=metadata["original_width"],
            original_height=metadata["original_height"],
            aspect_ratio=metadata["aspect_ratio"],
            channels=metadata["channels"],
            file_size_kb=metadata["file_size_kb"],
            sha256=metadata["sha256"],
            temp_path=str(saved_path),
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload processing failed: {exc}")


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Deep Learning"])
def predict_image(file: UploadFile = File(...)):
    """Classify chest radiograph and return predicted category and class probabilities."""
    try:
        contents = file.file.read()
        result = predictor.predict(contents)

        return PredictionResponse(
            predicted_category=result["predicted_category"],
            predicted_index=result["predicted_index"],
            confidence_score=result["confidence_score"],
            confidence_percentage=result["confidence_percentage"],
            class_probabilities=result["class_probabilities"],
            margin=result["margin"],
            inference_time_ms=result["inference_time_ms"],
            model_name=result["model_name"],
            device=result["device"],
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")


@app.post("/api/v1/gradcam", response_model=GradCAMResponse, tags=["Explainable AI"])
def generate_gradcam(
    file: UploadFile = File(...),
    colormap: str = Query("JET", pattern="^(JET|VIRIDIS|TURBO|MAGMA|PLASMA)$"),
    alpha: float = Query(0.45, ge=0.1, le=0.9),
):
    """Generate Grad-CAM activation heatmap and blended overlay."""
    try:
        cmap = colormap if isinstance(colormap, str) else "JET"
        a = float(alpha) if isinstance(alpha, (int, float)) else 0.45
        contents = file.file.read()
        pred_res = predictor.predict(contents)
        input_tensor = pred_res["input_tensor"]
        original_rgb = pred_res["original_rgb"]

        # Run Grad-CAM on target layer
        cam = GradCAM(predictor.model)
        try:
            heatmap_2d = cam.generate_heatmap(input_tensor, target_class_idx=pred_res["predicted_index"])
            colored_rgb, overlay_rgb = cam.overlay_heatmap(original_rgb, heatmap_2d, alpha=a, colormap=cmap)
        finally:
            cam.remove_hooks()

        analysis_id = generate_analysis_id()
        gradcam_save_path = settings.PLOTS_DIR / f"gradcam_{analysis_id}.png"
        cv2.imwrite(str(gradcam_save_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        b64_str = image_to_base64(overlay_rgb)
        method_info = GradCAM.explain_gradcam_educational()

        return GradCAMResponse(
            analysis_id=analysis_id,
            target_layer=settings.GRADCAM_TARGET_LAYER,
            colormap=cmap,
            gradcam_image_base64=b64_str,
            gradcam_image_path=str(gradcam_save_path),
            method_explanation=method_info,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Grad-CAM generation failed: {exc}")


@app.post("/api/v1/explain", response_model=ExplanationResponse, tags=["Generative AI"])
def generate_explanation(req: ExplanationRequest):
    """Generate structured educational explanation via Gemini LLM or scientific engine."""
    try:
        spatial_desc = req.spatial_description or "Diffuse activation across middle and lower pulmonary zones."
        explanation = gemini_client.generate_educational_explanation(
            predicted_category=req.predicted_category,
            confidence_score=req.confidence_score,
            normal_probability=req.normal_probability,
            pneumonia_probability=req.pneumonia_probability,
            model_name=req.model_name,
            gradcam_description=spatial_desc,
        )
        return ExplanationResponse(**explanation)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Explanation failed: {exc}")


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse, tags=["Pipeline"])
def run_full_analysis(file: UploadFile = File(...)):
    """
    End-to-End Orchestrated Pipeline:
    Upload -> Preprocess -> DenseNet121 Inference -> Grad-CAM -> Gemini LLM -> PDF Report -> SQLite
    """
    try:
        contents = file.file.read()
        # 1. Validation
        val_meta = ImagePreprocessor.validate_image_file(contents)
        analysis_id = generate_analysis_id()
        safe_name = sanitize_filename(file.filename or "xray.png")

        # Save original image
        raw_img_path = settings.UPLOADS_DIR / f"{analysis_id}_{safe_name}"
        with open(raw_img_path, "wb") as f:
            f.write(contents)

        # 2. Model Prediction
        pred = predictor.predict(contents)
        predicted_cat = pred["predicted_category"]
        confidence = pred["confidence_score"]
        normal_p = pred["class_probabilities"]["NORMAL"]
        pneumonia_p = pred["class_probabilities"]["PNEUMONIA"]
        latency = pred["inference_time_ms"]

        # 3. Grad-CAM Visualization
        cam = GradCAM(predictor.model)
        try:
            heatmap_2d = cam.generate_heatmap(pred["input_tensor"], target_class_idx=pred["predicted_index"])
            _, overlay_rgb = cam.overlay_heatmap(pred["original_rgb"], heatmap_2d)
        finally:
            cam.remove_hooks()

        gradcam_img_path = settings.PLOTS_DIR / f"gradcam_{analysis_id}.png"
        cv2.imwrite(str(gradcam_img_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        # 4. Educational LLM Narrative
        narrative_res = report_generator.generate_full_explanation(pred, heatmap_2d)
        ai_explanation = narrative_res["educational_explanation"]

        # 5. PDF Generation
        pdf_path = pdf_generator.generate_report(
            analysis_id=analysis_id,
            original_image_path=raw_img_path,
            gradcam_image_path=gradcam_img_path,
            prediction_data=pred,
            ai_explanation=ai_explanation,
        )

        # 6. SQLite Record Persistence
        db.save_analysis({
            "analysis_id": analysis_id,
            "image_hash": val_meta["sha256"],
            "image_filename": safe_name,
            "image_path": str(raw_img_path),
            "predicted_category": predicted_cat,
            "confidence_score": confidence,
            "normal_probability": normal_p,
            "pneumonia_probability": pneumonia_p,
            "model_name": pred["model_name"],
            "model_version": settings.VERSION,
            "inference_time_ms": latency,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "gradcam_image_path": str(gradcam_img_path),
            "pdf_report_path": str(pdf_path),
            "ai_explanation_json": ai_explanation,
            "notes": "Full end-to-end API pipeline execution",
        })

        return FullAnalysisResponse(
            analysis_id=analysis_id,
            image_filename=safe_name,
            image_hash=val_meta["sha256"],
            predicted_category=predicted_cat,
            confidence_score=confidence,
            normal_probability=normal_p,
            pneumonia_probability=pneumonia_p,
            inference_time_ms=latency,
            model_name=pred["model_name"],
            model_version=settings.VERSION,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            gradcam_image_path=str(gradcam_img_path),
            pdf_report_path=str(pdf_path),
            educational_explanation=ai_explanation,
            disclaimer=settings.DISCLAIMER_TEXT,
        )

    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Full analysis failed: {exc}")


@app.get("/api/v1/history", response_model=HistoryListResponse, tags=["History & Records"])
def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str = Query("", description="Search by ID or filename"),
    category: str = Query("", description="Filter by category (NORMAL or PNEUMONIA)"),
):
    """Query analysis history with pagination, keyword search, and class filtering."""
    lim = limit if isinstance(limit, int) else 50
    off = offset if isinstance(offset, int) else 0
    s = search if isinstance(search, str) else ""
    cat = category if isinstance(category, str) else ""
    records = db.list_analyses(limit=lim, offset=off, search_query=s, category=cat)
    return HistoryListResponse(total_count=len(records), records=records)


@app.get("/api/v1/history/{analysis_id}", tags=["History & Records"])
def get_analysis_detail(analysis_id: str):
    """Retrieve full analysis detail by ID."""
    record = db.get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis ID '{analysis_id}' not found.")
    return record


@app.delete("/api/v1/history/{analysis_id}", tags=["History & Records"])
def delete_analysis_record(analysis_id: str, delete_files: bool = Query(True)):
    """Delete an analysis record and its artifacts from storage."""
    success = db.delete_analysis(analysis_id, delete_files=delete_files)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis ID '{analysis_id}' not found.")
    return {"status": "success", "message": f"Analysis '{analysis_id}' successfully removed."}


@app.get("/api/v1/reports/{analysis_id}/pdf", tags=["Reports"])
def download_pdf_report(analysis_id: str):
    """Download generated ReportLab PDF clinical research report."""
    record = db.get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis '{analysis_id}' not found.")

    pdf_path = record.get("pdf_report_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF report file does not exist on disk.")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"MediVision_Report_{analysis_id}.pdf",
    )


@app.get("/api/v1/dataset/summary", response_model=DatasetSummaryResponse, tags=["Dataset"])
def get_dataset_summary():
    """Returns local dataset scanning status, class counts, and split summaries."""
    summary = dataset_manager.get_dataset_summary()
    return DatasetSummaryResponse(
        dataset_available=summary["dataset_available"],
        grand_total=summary["grand_total"],
        splits=summary["splits"],
        class_distribution=summary["class_distribution"],
        imbalance_ratio=summary["imbalance_ratio"],
        instructions=summary["instructions"],
    )
