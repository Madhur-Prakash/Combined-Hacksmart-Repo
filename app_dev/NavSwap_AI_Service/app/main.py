from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import os
import google.generativeai as genai
from .schemas import (
    StationMetrics, LoadPredictionResponse, FaultPredictionResponse, 
    ActionPredictionResponse, ExplanationRequest, ExplanationResponse
)
from .prediction import model_manager

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Gemini setup - User needs to ensure API key is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    model_manager.load_models()
    yield
    # Clean up (if needed)

app = FastAPI(
    title="NavSwap AI Prediction Microservice",
    description="API for EV Battery Management AI Models",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": model_manager.xgb_queue_model is not None}

@app.post("/predict-load", response_model=LoadPredictionResponse)
async def predict_load(metrics: StationMetrics):
    try:
        queue, wait = model_manager.predict_load(metrics)
        return LoadPredictionResponse(
            predicted_queue_length=float(queue),
            predicted_avg_wait_time=float(wait)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-fault", response_model=FaultPredictionResponse)
async def predict_fault(metrics: StationMetrics):
    try:
        risk, prob = model_manager.predict_fault(metrics)
        return FaultPredictionResponse(
            fault_risk=int(risk),
            fault_probability=float(prob)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-action", response_model=ActionPredictionResponse)
async def predict_action(metrics: StationMetrics):
    try:
        encoded, label, proba = model_manager.predict_action(metrics)
        return ActionPredictionResponse(
            action_encoded=int(encoded),
            action_label=label,
            action_probabilities=proba
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-decision", response_model=ExplanationResponse)
async def explain_decision(request: ExplanationRequest = Body(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API Key not configured")
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Explain the following AI decision for an EV station:
        Action: {request.action}
        Queue Prediction: {request.queue_prediction} vehicles
        Wait Time: {request.wait_time} minutes
        Fault Probability: {request.fault_probability:.2f}
        Reliability Score: {request.station_reliability}
        
        Provide a concise, human-readable explanation for a station operator.
        """
        response = model.generate_content(prompt)
        return ExplanationResponse(explanation=response.text)
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
