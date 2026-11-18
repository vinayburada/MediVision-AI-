# MediVision AI - Dockerfile
# Production-ready multi-stage container build for educational medical image analysis

FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8501

WORKDIR $APP_HOME

# Install system dependencies for OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Ensure storage directories exist
RUN mkdir -p data/raw/uploads data/processed data/train data/val data/test data/samples \
    artifacts/checkpoints artifacts/plots artifacts/reports database

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Default command: Launch Streamlit web dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
