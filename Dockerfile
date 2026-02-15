# ============================================================================
# EuroSAT Classification — Docker Build (Single Stage, Self-Contained)
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY api/ api/
COPY public/ public/

# Create directories
RUN mkdir -p /app/checkpoints /app/outputs /app/logs

# Download model checkpoint from GitHub (self-contained, no upload dependency)
RUN python -c "\
    import urllib.request, os; \
    url = 'https://github.com/Ahmad-Abudllah-Ahmad/sparkai.ae-task-for-job/raw/main/checkpoints/best_baseline.pth'; \
    dest = '/app/checkpoints/best_baseline.pth'; \
    print(f'Downloading model from {url}...'); \
    urllib.request.urlretrieve(url, dest); \
    size = os.path.getsize(dest); \
    print(f'Downloaded {size} bytes to {dest}')"

# Environment variables
ENV CONFIG_PATH=/app/configs/config.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run inference API
CMD sh -c "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"
