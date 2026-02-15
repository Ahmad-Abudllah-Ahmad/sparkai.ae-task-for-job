# ============================================================================
# EuroSAT Classification — Docker Multi-Stage Build
# ============================================================================
# Stage 1: Base with dependencies
# Stage 2: Slim inference runtime
# ============================================================================

# --- Stage 1: Build ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY api/ api/
COPY public/ public/

# Create directories for model and outputs
RUN mkdir -p /app/checkpoints /app/outputs /app/logs

# Environment variables
ENV CONFIG_PATH=/app/configs/config.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default: run inference API (via Vercel adapter for frontend support)
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
