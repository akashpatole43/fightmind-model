# ─────────────────────────────────────────
# FightMind AI — Python Model Service
# Dockerfile
# ─────────────────────────────────────────

# Use slim Python image (smaller + faster builds)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only re-installs when requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create directories that need to exist at runtime
RUN mkdir -p data/raw data/processed embeddings/vectorstore models/fine_tuned

# Expose FastAPI port
EXPOSE 8000

# Health check (used by docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
