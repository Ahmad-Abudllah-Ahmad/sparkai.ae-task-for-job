"""
FastAPI Inference Server
========================
REST API for EuroSAT satellite image classification.

Endpoints:
    GET  /health   — Health check
    POST /predict  — Classify an uploaded image
    GET  /info     — Model and class information
"""

import io
import logging
import os
import sys
import time
from typing import Dict

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from src.inference.pipeline import InferencePipeline, CLASS_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EuroSAT Classification API",
    description=(
        "REST API for satellite image land-use classification. "
        "Classifies Sentinel-2 satellite imagery into 10 land-use categories."
    ),
    version="1.0.0",
)

# Global pipeline — loaded on startup
_pipeline: InferencePipeline = None


def _load_pipeline() -> InferencePipeline:
    """Load inference pipeline from config."""
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return InferencePipeline.from_config(config)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    global _pipeline
    try:
        _pipeline = _load_pipeline()
        logger.info("Inference pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.warning("Server started without model — /predict will fail.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _pipeline is not None,
        "timestamp": time.time(),
    }


@app.get("/info")
async def model_info():
    """Return model and class information."""
    return {
        "model": "EuroSAT Land-Use Classifier",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "description": (
            "Classifies Sentinel-2 satellite imagery into "
            "10 land-use / land-cover categories."
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Classify an uploaded satellite image.

    Args:
        file: Image file (JPEG, PNG, TIFF)

    Returns:
        JSON with predicted class, confidence, and all probabilities
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Accepted: {', '.join(allowed_types)}"
        )

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        start_time = time.time()
        result = _pipeline.predict(image)
        inference_time = time.time() - start_time

        result["inference_time_ms"] = round(inference_time * 1000, 2)
        result["filename"] = file.filename

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Run directly with: python -m src.inference.api
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
