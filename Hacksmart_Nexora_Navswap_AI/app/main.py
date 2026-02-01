from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routes.predict import router as predict_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NavSwap AI Prediction Microservice",
    description="AI Analytics Layer for Smart EV Battery Swap Management System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "NavSwap AI Prediction Microservice", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "navswap-ai-prediction"}

if __name__ == "__main__":
    import uvicorn
    from app.utils.config import HOST, PORT
    
    logger.info("Starting NavSwap AI Prediction Microservice...")
    uvicorn.run(app, host=HOST, port=PORT)