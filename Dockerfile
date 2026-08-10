# Hugging Face Spaces Docker build for MoodSyncAI.
# Multi-stage build: install deps once, pre-cache model weights at build time,
# then a lean runtime layer.

FROM python:3.11-slim AS base

# System libs needed by opencv-headless, librosa, mediapipe, ffmpeg, whisper.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for cache-friendly layering.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-cache HF + Whisper checkpoints at build time so cold start is instant.
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

COPY moodsync ./moodsync
COPY scripts ./scripts
RUN python -m scripts.download_models

# Bring in the rest of the project.
COPY . .

# HuggingFace Spaces expects the app to listen on 7860; Streamlit Cloud uses 8501.
# Both are accepted via the PORT env var convention.
ENV PORT=7860
EXPOSE 7860 8501

# Streamlit specifics.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["bash", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
