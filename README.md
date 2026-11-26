# 🩺 MediVision AI: Intelligent Medical Image Analysis & Explainable AI Report Generation

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20CPU%2FCUDA-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.163-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io/)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-purple.svg)](https://arxiv.org/abs/1610.02391)
[![Generative AI](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-31%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Educational%20Research-lightgrey.svg)]()

> **⚠️ MANDATORY EDUCATIONAL RESEARCH DISCLAIMER:**  
> MediVision AI is an experimental computer vision, explainable AI, and educational demonstration system developed strictly for academic, educational, and scientific research purposes. This software is **NOT** a certified medical diagnostic system and has **NOT** been evaluated or cleared by the US FDA, CE, CDSCO, or any international medical regulatory agency. It must **NEVER** be used to formulate medical diagnoses, establish clinical opinions, or direct patient treatment decisions. Always consult a licensed radiologist or healthcare professional for diagnostic care.



## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Architecture & Features](#-key-architecture--features)
3. [Technology Stack](#-technology-stack)
4. [Dataset & Patient Leakage Prevention](#-dataset--patient-leakage-prevention)
5. [System Architecture Diagram](#-system-architecture-diagram)
6. [Folder Structure](#-complete-folder-structure)
7. [Installation & Setup](#-installation--setup)
8. [Running the Application](#-running-the-application)
9. [FastAPI REST API Documentation](#-fastapi-rest-api-documentation)
10. [Model Training & Evaluation](#-model-training--evaluation)
11. [Explainable AI (Grad-CAM) Implementation](#-explainable-ai-grad-cam)
12. [Generative AI & Structured Reporting](#-generative-ai--structured-reporting)
13. [PDF Report Generation](#-pdf-report-generation)
14. [Automated Testing Suite](#-automated-testing-suite)
15. [Git Commands for Publishing](#-git-commands-for-publishing)
16. [Limitations & Bioethical Considerations](#-limitations--bioethical-considerations)

---

## 🎯 Project Overview

Deep learning models applied to medical radiography often suffer from a severe "black-box" dilemma: neural networks can achieve high classification accuracy on benchmarks by exploiting non-pathological confounders (e.g., hospital acquisition tags, patient posture, or scanner contrast variations) rather than true clinical signs of infection.

**MediVision AI** resolves this interpretability gap through an integrated end-to-end framework:
- **Convolutional Neural Network (DenseNet121):** Deep feature reuse for binary chest X-ray classification (`NORMAL` vs `PNEUMONIA`).
- **Vision Transformer (ViT-B/16):** Global self-attention benchmark comparing patch-level representations with CNNs.
- **Explainable AI (Grad-CAM):** Mathematical gradient-weighted activation mapping identifying penultimate layer spatial attention.
- **Generative AI (Google Gemini):** Multi-modal LLM reasoning translating raw probabilities and spatial attention patterns into structured educational narratives.
- **Publication-Grade Reporting:** Automated ReportLab PDF generator compiling side-by-side radiograph overlays, metadata tables, and clinical disclaimers.
---

## 🚀 Key Architecture & Features

| Capability | Module | Highlights |
| :--- | :--- | :--- |
| **OpenCV Preprocessing** | `preprocessing/` | Grayscale/RGB validation, Bicubic resizing (224×224), ImageNet standardization, corrupted file rejection, data hygiene. |
| **Data Leakage Prevention** | `preprocessing/image_loader.py` | Extracts patient IDs (`person1946`, `IM-0378`) to enforce patient-level isolation between train, validation, and test sets. |
| **Deep Learning Models** | `models/` | Transfer learning with PyTorch `DenseNet121` (7.05M params) and `ViT-B/16` (86M params) with early stopping and class-weighted Cross-Entropy. |
| **Visual Explainability** | `explainability/gradcam.py` | PyTorch hooks capturing gradients from `features.denseblock4.denselayer16.conv2`, bilinear upsampling, JET/VIRIDIS/TURBO colormaps. |
| **Generative AI Reasoning** | `llm/` | Google Gemini API integration with strict JSON output schemas and offline deterministic research fallbacks when unconfigured. |
| **Audit Database** | `database/` | Thread-safe SQLite management with SHA-256 image hashes, parameterized queries, search filtering, and deletion cascading. |
| **Interactive Dashboard** | `app.py` & `pages/` | Multi-page Streamlit UI in clinical Navy Blue (`#0F2942`) and Teal (`#00A896`), interactive Plotly charts, and sample selectors. |
| **Production REST API** | `api/` | Modular FastAPI service with Pydantic request/response schemas, OpenAPI/Swagger docs, and synchronous ML offloading. |

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.11 / 3.12 / 3.13 (Object-Oriented, Type-Annotated)
- **Deep Learning:** PyTorch 2.15+, TorchVision 0.29+
- **Computer Vision:** OpenCV 5.1+, Pillow 13.0+
- **Machine Learning & Analytics:** Scikit-Learn 1.9+, Pandas 3.1+, NumPy 2.5+, SciPy 1.18+
- **Interactive Visualizations:** Plotly 7.1+, Matplotlib 3.11+, Seaborn 0.13+
- **Backend REST API:** FastAPI 0.163+, Uvicorn 0.54+, Pydantic 2.14+
- **Frontend Dashboard:** Streamlit 1.67+
- **Report Generation:** ReportLab 5.1+ (Platypus two-pass canvas)
- **Generative AI:** Google GenAI / Google Generative AI SDK, JSON Prompt Engineering
- **Database:** SQLite 3 (Parameterized SQL)
- **Quality Assurance:** Pytest 9.2+

---

## 📁 Complete Folder Structure

```
MediVision-AI/
├── app.py                      # Streamlit Home Dashboard & KPI Overview
├── requirements.txt            # Pinned, cross-compatible dependencies
├── README.md                   # Comprehensive scientific & technical documentation
├── .env.example                # Template for Gemini API key & configuration
├── .gitignore                  # Git ignore rules for checkpoints, data, environments
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Multi-container orchestration (Streamlit + FastAPI)
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Global constants, paths, color palette, and disclaimers
│
├── data/                       # Dataset directories (kept with .gitkeep)
│   ├── raw/
│   │   └── uploads/            # Temporary storage for uploaded radiographs
│   ├── processed/              # Normalized tensors and caches
│   ├── train/                  # Training split (NORMAL / PNEUMONIA)
│   ├── val/                    # Validation split (patient-isolated)
│   ├── test/                   # Unseen test split
│   └── samples/                # Built-in synthetic educational chest radiograph phantoms
│
├── models/
│   ├── __init__.py
│   ├── architecture.py         # DenseNet121 & Vision Transformer classifier heads
│   ├── train.py                # Patient-level split, class weighting, early stopping
│   ├── predict.py              # Inference engine, probability estimation, latency profiling
│   └── model_utils.py          # Metric computation, checkpoint serialization, hardware detection
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_preprocessing.py  # OpenCV standardization, ImageNet norm, corruption checks
│   ├── image_loader.py         # Dataset scanner, patient ID parsing, EDA summary generator
│   └── augmentation.py         # Clinically sound augmentations (no inverted anatomies)
│
├── explainability/
│   ├── __init__.py
│   └── gradcam.py              # Grad-CAM PyTorch forward/backward hooks & colormap overlays
│
├── llm/
│   ├── __init__.py
│   ├── gemini_client.py        # Google Gemini API connector with offline fallback template
│   └── report_generator.py     # Hotspot spatial quadrant analysis & report coordination
│
├── database/
│   ├── __init__.py
│   ├── db.py                   # SQLite connection, schema initialization, parameterized queries
│   ├── history.py              # Service layer for UI filtering, search, and audit tables
│   └── medivision.db           # SQLite database file
│
├── dashboard/
│   ├── __init__.py
│   └── analytics.py            # Plotly donut charts, histograms, confusion matrices, ROC curves
│
├── utils/
│   ├── __init__.py
│   ├── pdf_generator.py        # ReportLab PDF generator with running header/footer
│   ├── helpers.py              # UUID formatting, Base64 converters, date formatters
│   └── ui_components.py        # Streamlit medical CSS, hero banners, badges, and cards
│
├── pages/
│   ├── 1_Image_Analysis.py     # Upload, live classification, Grad-CAM, and PDF download
│   ├── 2_Explainable_AI.py     # Opacity sliders, colormaps (JET, VIRIDIS, TURBO), layer analysis
│   ├── 3_AI_Explanation.py     # Gemini LLM findings, uncertainty interpretations, prompt inspection
│   ├── 4_Analytics_Dashboard.py# Evaluated model metrics, confusion matrix, DenseNet vs ViT
│   ├── 5_Analysis_History.py   # Search, date/category filters, record inspection, PDF re-download
│   └── 6_About_Project.py      # Architecture documentation, technology breakdown, ethics
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI route endpoints (/health, /upload, /predict, /analyze)
│   └── schemas.py              # Pydantic input/output schemas
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Image validation, tensor dimensions, patient extraction
│   ├── test_prediction.py      # Forward passes, softmax probabilities, metric functions
│   ├── test_gradcam.py         # Hook cleanup, heatmap bounds, colormap blending
│   ├── test_database.py        # SQLite queries, filters, record deletion, KPI counters
│   ├── test_llm.py             # Fallback generator, spatial descriptions, safety disclaimers
│   ├── test_pdf.py             # PDF generation, Platypus flowables, file integrity
│   └── test_api.py             # FastAPI endpoint verification and error responses
│
└── artifacts/
    ├── checkpoints/            # Trained weights (densenet121_best.pth)
    ├── plots/                  # Training history curves & Grad-CAM outputs
    └── reports/                # Generated research PDF publications
```

---

## 🔬 Dataset & Patient Leakage Prevention

### Sourcing
MediVision AI is built around the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney:
- **Kaggle Source:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL` and `PNEUMONIA` (Bacterial & Viral)
- **Volume:** 5,856 Anterior-Posterior Pediatric Chest Radiographs

### Preventing Patient-Level Data Leakage
Standard random splitting of medical images leads to **severe patient leakage**: multiple images from the same patient (e.g. follow-up radiographs) end up in both training and testing sets. The model memorizes patient-specific thoracic morphology rather than learning pathological opacities, producing falsely inflated test performance.

**MediVision AI's Implementation:**
1. The `DatasetManager` extracts patient identifiers using regular expressions:
   - Pneumonia: `person1946_bacteria_4874.jpeg` $\rightarrow$ Patient ID: `person1946`
   - Normal: `NORMAL2-IM-0378-0001.jpeg` $\rightarrow$ Patient ID: `normal2-im-0378`
2. Stratification and partitioning group images by unique `patient_id`.
3. If the Kaggle dataset is not yet downloaded, MediVision AI automatically generates **synthetic educational chest radiograph phantoms** with realistic thoracic ribs, lung fields, cardiac silhouettes, and consolidation patterns so the application runs immediately out-of-the-box.

---

## 📐 System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|   Streamlit Multi-Page Web App (Dashboard, Analysis, XAI, Gemini LLM, History)    |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               REST API / BACKEND                                  |
|   FastAPI REST Microservice (/health, /upload, /predict, /gradcam, /analyze)      |
+-------------------+--------------------+--------------------+---------------------+
                    |                    |                    |
                    v                    v                    v
+-----------------------+  +-----------------------+  +-----------------------------+
|  PREPROCESSING ENGINE |  | DEEP LEARNING (PyTorch)|  |    EXPLAINABLE AI (XAI)     |
| - OpenCV Bicubic (224)|  | - DenseNet121 CNN     |  | - Backward Hooks on Conv2   |
| - ImageNet Norm (μ,σ) |  | - ViT-B/16 Transformer|  | - Global Average Pooling    |
| - Corrupt File Reject |  | - Softmax Probability |  | - Blended Colormap Overlays |
+-----------------------+  +-----------------------+  +-----------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               GENERATIVE AI LAYER                                 |
|   Google Gemini API (Structured JSON Prompts, Uncertainty & Educational Findings) |
|   * Automatic deterministic fallback if API key is not configured                 |
+-------------------+--------------------+--------------------+---------------------+
                    |                    |                    |
                    v                    v                    v
+-----------------------+  +-----------------------+  +-----------------------------+
|    AUDIT DATABASE     |  |     PDF GENERATOR     |  |     ANALYTICS ENGINE        |
| - SQLite3 Persistence |  | - ReportLab Platypus  |  | - Plotly Donut & Bar Charts |
| - SHA-256 Hashes      |  | - Side-by-side Layout |  | - Confusion Matrix & ROC    |
| - Parameterized Query |  | - Running Headers     |  | - Latency & Confidence Dist |
+-----------------------+  +-----------------------+  +-----------------------------+
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git
- Windows, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MediVision-AI.git
cd MediVision-AI
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` in any editor:
```env
# Google Gemini API Key (optional for basic vision model, required for live LLM explanations)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# App settings
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
```
> **Note:** MediVision AI functions **100% locally** without a paid API key for image preprocessing, DenseNet121 classification, Grad-CAM overlays, SQLite persistence, and PDF report generation. When `GEMINI_API_KEY` is not provided, the system seamlessly uses its verified scientific educational fallback engine.

---

## 🚀 Running the Application

### Option A: Launch the Streamlit Frontend (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch the FastAPI REST Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive OpenAPI / Swagger documentation at `http://localhost:8000/docs`.

### Option C: Run via Docker Compose
```bash
docker-compose up --build
```
- Streamlit Web Dashboard: `http://localhost:8501`
- FastAPI Documentation: `http://localhost:8000/docs`

---

## 📡 FastAPI REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Returns runtime health, PyTorch version, hardware compute device, and SQLite connection status. |
| `POST` | `/api/v1/upload` | Validates image header, formats, dimensions, and saves with SHA-256 hash. |
| `POST` | `/api/v1/predict` | Executes deep learning forward pass; returns predicted class, confidence, and probabilities. |
| `POST` | `/api/v1/gradcam` | Extracts penultimate layer gradients and returns Base64-encoded activation overlay. |
| `POST` | `/api/v1/explain` | Generates structured educational LLM narrative from prediction parameters. |
| `POST` | `/api/v1/analyze` | End-to-end pipeline (Upload $\rightarrow$ Predict $\rightarrow$ Grad-CAM $\rightarrow$ LLM $\rightarrow$ PDF $\rightarrow$ SQLite). |
| `GET` | `/api/v1/history` | Paginated search of analysis records with keyword and category filtering. |
| `GET` | `/api/v1/history/{id}` | Retrieves complete details for a specific analysis identifier. |
| `DELETE`| `/api/v1/history/{id}` | Deletes record from SQLite and purges associated image and PDF files from disk. |
| `GET` | `/api/v1/reports/{id}/pdf` | Downloads the generated ReportLab clinical research PDF report. |
| `GET` | `/api/v1/dataset/summary`| Returns dataset distribution, split counts, and availability status. |

---

## 🏋️ Model Training & Evaluation

To train the DenseNet121 model using patient-level stratification, class weighting, and early stopping:

```bash
# Full training on local dataset
python -m models.train --model densenet121 --epochs 10 --batch-size 16 --lr 0.0001

# Fast smoke-test mode (verifies pipeline and saves valid checkpoint)
python -m models.train --smoke-test --epochs 2 --batch-size 4
```

### Evaluation Metrics
MediVision AI calculates real mathematical metrics without fabrication:
- **Accuracy:** Overall ratio of correct classifications across all classes.
- **Precision:** $\frac{TP}{TP + FP}$ — Clinical positive predictive value.
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Critical for minimizing missed pneumonia cases.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under the receiver operating characteristic curve.
- **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives.

Saved model checkpoints and training history CSVs are stored in `artifacts/checkpoints/` and `artifacts/plots/`.

---

## 🔍 Explainable AI (Grad-CAM)

Gradient-weighted Class Activation Mapping calculates the gradients of the winning category score $y^c$ with respect to feature maps $A^k$ in DenseNet121's penultimate layer (`features.denseblock4.denselayer16.conv2`):

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

- **Rectified Linear Unit (ReLU):** Filters out features that contribute negatively to the target class, focusing exclusively on positive evidence.
- **Colormaps:** Supports `JET`, `VIRIDIS`, `TURBO`, `MAGMA`, and `PLASMA` with user-configurable alpha blending opacity ($\alpha = 0.1 \dots 0.9$).
- **Interpretability Disclaimer:** Highlights mathematical pixel correlations influencing layer weights; does not establish definitive microbiological or pathological causality.

---

## ✨ Generative AI & Structured Reporting

Google Gemini API integration uses strict prompt engineering to generate standardized JSON structures:

```json
{
  "title": "AI-GENERATED EDUCATIONAL EXPLANATION",
  "source": "Google Gemini API (Live)",
  "executive_summary": "High-level interpretation of the model classification.",
  "technical_findings": "Detailed breakdown of the probability distribution.",
  "gradcam_interpretation": "Spatial description of where activations clustered.",
  "model_limitations": "Architecture boundaries and false positive risks.",
  "dataset_limitations": "Pediatric single-center demographic bias notes.",
  "educational_disclaimer": "Prominent non-diagnostic warning statement."
}
```

### Safety Guardrails
1. **Never Diagnoses:** Refuses requests to make a medical diagnosis or triage.
2. **Never Prescribes:** Does not generate medication, dosing, or therapeutic suggestions.
3. **Never Hallucinates:** Bases all commentary strictly on model probabilities and spatial coordinates.

---

## 📄 PDF Report Generation

MediVision AI generates downloadable research reports using ReportLab Platypus:
- **Header:** Professional MediVision AI branding with UTC generation timestamp.
- **Audit Table:** Unique Analysis ID (`MVA-YYYYMMDD-XXXX`), Model Architecture, Device, Latency.
- **Classification Scores:** Category badges, Normal vs Pneumonia percentage bars.
- **Side-by-Side Visuals:** Original frontal chest radiograph aligned with the Grad-CAM activation heatmap.
- **Educational Narrative:** Executive summary, technical analysis, and spatial findings.
- **Mandatory Boxed Disclaimer:** High-visibility alert framing the document as experimental research only.
- **Two-Pass Numbered Canvas:** Dynamic `"Page X of Y"` pagination.

---

## 🧪 Automated Testing Suite

The project includes 31 unit and integration tests executing across all system modules:

```bash
# Run the entire test suite
pytest tests/ -p no:asyncio -v
```

### Test Suite Summary
```
tests/test_api.py::test_api_health PASSED
tests/test_api.py::test_api_upload_valid PASSED
tests/test_api.py::test_api_upload_invalid PASSED
tests/test_api.py::test_api_predict PASSED
tests/test_api.py::test_api_gradcam PASSED
tests/test_api.py::test_api_explain PASSED
tests/test_api.py::test_api_full_analysis PASSED
tests/test_api.py::test_api_dataset_summary PASSED
tests/test_api.py::test_api_history PASSED
tests/test_database.py::test_db_init_and_save PASSED
tests/test_database.py::test_list_and_filter PASSED
tests/test_database.py::test_delete_analysis PASSED
tests/test_database.py::test_statistics PASSED
tests/test_gradcam.py::test_gradcam_hook_and_generation PASSED
tests/test_gradcam.py::test_gradcam_overlay PASSED
tests/test_gradcam.py::test_explain_gradcam_educational PASSED
tests/test_llm.py::test_gemini_client_fallback_mode PASSED
tests/test_llm.py::test_report_generator_hotspots PASSED
tests/test_llm.py::test_report_generator_full PASSED
tests/test_pdf.py::test_pdf_report_generation PASSED
tests/test_prediction.py::test_densenet_architecture_forward PASSED
tests/test_prediction.py::test_vit_architecture_forward PASSED
tests/test_prediction.py::test_predictor_inference PASSED
tests/test_prediction.py::test_calculate_metrics PASSED
tests/test_preprocessing.py::test_validate_image_file_valid PASSED
tests/test_preprocessing.py::test_validate_image_file_corrupt PASSED
tests/test_preprocessing.py::test_validate_image_file_empty PASSED
tests/test_preprocessing.py::test_load_and_preprocess_tensor_shape PASSED
tests/test_preprocessing.py::test_denormalize_for_display PASSED
tests/test_preprocessing.py::test_extract_patient_id PASSED
tests/test_preprocessing.py::test_opencv_train_aug PASSED

============================= 31 passed in 15.47s =============================
```

---

## 🐙 Git Commands for Publishing

To publish the complete project to GitHub:

```bash
# 1. Initialize git repository
git init

# 2. Stage all source files (respecting .gitignore)
git add .

# 3. Verify that large datasets, private keys, and .venv are ignored
git status

# 4. Commit code
git commit -m "feat: complete end-to-end MediVision AI medical imaging and XAI platform"

# 5. Set main branch and link remote repository
git branch -M main
git remote add origin https://github.com/<your-username>/MediVision-AI.git

# 6. Push to GitHub
git push -u origin main
```
---

## ⚖️ Limitations & Bioethical Considerations

1. **Non-Diagnostic Scope:** This application is an educational prototype. It is not licensed for medical diagnostics and must never be used in hospital workflows or clinical patient care.
2. **Demographic & Pediatric Bias:** The primary training data is derived from pediatric patients aged 1 to 5. Features learned may not generalize to adult or elderly patient cohorts without domain adaptation.
3. **Binary Classification Constraint:** The system is constrained to binary classification (`NORMAL` vs `PNEUMONIA`) and cannot detect pleural effusions, pneumothorax, pulmonary tuberculosis, nodules, or cardiomegaly.
4. **Grad-CAM Limitations:** Heatmaps highlight mathematical feature sensitivity, which may occasionally latch onto non-pathological artifacts such as chest tube hardware, medical monitoring clips, or high-contrast rib edges.
5. **Patient Privacy & HIPAA:** MediVision AI does not collect or store patient names, MRNs, or sensitive identifiers. Records are tracked purely via anonymous SHA-256 checksums and internal identifiers.
