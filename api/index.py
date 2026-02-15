import os
import sys
import logging
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import yaml

# Add project root to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.pipeline import InferencePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EuroSAT Vercel API")


# Global variables to hold pipelines
pipelines: Dict[str, InferencePipeline] = {}
config: Dict[str, Any] = {}

def load_config():
    global config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
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
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_baseline.pth')
        else: # resnet
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_resnet.pth')
            
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
        
        result = pipeline.predict(image)
        result["model_used"] = model_type
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount public directory for static files (frontend) settings
# We mount at root to simulate Vercel's behavior where everything not /api is static
# API routes defined above take precedence
public_path = os.path.join(os.path.dirname(__file__), '..', 'public')
if os.path.exists(public_path):
    app.mount("/", StaticFiles(directory=public_path, html=True), name="public")
