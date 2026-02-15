import os
import sys
import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import yaml

# Add project root to path to import from src
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.inference.pipeline import InferencePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EuroSAT Vercel API")


# Global variables to hold pipelines
pipelines: Dict[str, InferencePipeline] = {}
config: Dict[str, Any] = {}


@app.on_event("startup")
async def preload_models():
    """Preload both models at startup to eliminate cold-start latency."""
    import torch
    logger.info("Preloading models at startup...")
    for model_type in ["baseline", "resnet"]:
        try:
            pipeline = get_pipeline(model_type)
            # Warmup pass: run a dummy tensor through the model
            # This triggers PyTorch internal optimizations (memory alloc, kernel selection)
            dummy = torch.randn(1, 3, 64, 64)
            with torch.inference_mode():
                pipeline.model(dummy.to(pipeline.device))
            logger.info(f"  {model_type} preloaded + warmed up")
        except Exception as e:
            logger.warning(f"  {model_type} preload failed: {e}")
    logger.info("Model preloading complete.")

def load_config():
    global config
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

def get_pipeline(model_type: str) -> InferencePipeline:
    """Lazy load pipeline for the specific model type."""
    if model_type not in pipelines:
        logger.info(f"Loading pipeline for model: {model_type}")
        
        # Create a copy of config to modify model type
        # We need to reload config to ensure we have a fresh copy if needed, 
        # but for now we just modify the global one's copy? 
        # Better to load fresh or copy deep.
        
        if not config:
            load_config()
            
        model_config = config.copy()
        model_config['model']['type'] = model_type
        
        # Set checkpoint path based on model type
        # Assuming checkpoints are named consistently
        if model_type == 'baseline':
            ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_baseline.pth')
        else: # resnet
            ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_resnet.pth')
            
        # Update config inference path so pipeline loads correct model
        if 'inference' not in model_config:
            model_config['inference'] = {}
        model_config['inference']['model_path'] = ckpt_path
        
        # Check if checkpoint exists
        if not os.path.exists(ckpt_path):
             logger.warning(f"Checkpoint not found for {model_type} at {ckpt_path}. Inference will likely fail.")
             # We let it proceed, pipeline might fail or init random model if allowed?
             # InferencePipeline.from_config usually loads state dict.
        
        try:
            pipelines[model_type] = InferencePipeline.from_config(model_config)
        except Exception as e:
            logger.error(f"Failed to load {model_type} pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_type}: {str(e)}")
            
    return pipelines[model_type]

@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": list(pipelines.keys())}

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("baseline") # Default to baseline
):
    """
    Predict using the specified model type (baseline or resnet).
    """
    allowed_models = {"baseline", "resnet"}
    if model_type not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Allowed: {allowed_models}")
    
    # Validate file
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff", "image/webp"]:
         raise HTTPException(status_code=400, detail="Invalid file type. Only images allowed.")

    try:
        pipeline = get_pipeline(model_type)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        result = pipeline.predict(image)
        inference_ms = round((time.time() - start_time) * 1000, 1)
        
        result["model_used"] = model_type
        result["inference_time_ms"] = inference_ms
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount public directory for static files (frontend) settings
# We mount at root to simulate Vercel's behavior where everything not /api is static
# API routes defined above take precedence
public_path = os.path.join(PROJECT_ROOT, 'public')
if os.path.exists(public_path):
    app.mount("/", StaticFiles(directory=public_path, html=True), name="public")
