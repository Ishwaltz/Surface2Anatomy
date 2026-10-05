# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Python backend dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    python-multipart==0.0.9 \
    pydantic==2.7.4 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    scikit-learn==1.5.0 \
    trimesh==4.4.1 \
    open3d==0.18.0 \
    httpx==0.27.0

# Copy application files
COPY api/ /app/api/
COPY sharon/ /app/sharon/
COPY tools/ /app/tools/
COPY experiments/phase10R/checkpoints/ /app/experiments/phase10R/checkpoints/
COPY experiments/phase16_brain/checkpoints/ /app/experiments/phase16_brain/checkpoints/
COPY tests/ /app/tests/

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
