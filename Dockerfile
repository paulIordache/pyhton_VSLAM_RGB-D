FROM python:3.11-slim

# Set environment variables to avoid Python writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 1) Install system dependencies in one optimized layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libeigen3-dev \
    libsuitesparse-dev \
    libxext6 \
    libxrender1 \
    libsm6 \
    libglib2.0-0 \
    x11-utils \
    xauth \
    python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2) Set working directory and install Python dependencies
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir g2o-python

# 3) Copy app source code
COPY . .

# 4) Default command
ENTRYPOINT ["python", "-u", "run_slam.py"]
