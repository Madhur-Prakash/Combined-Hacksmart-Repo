import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model paths
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "models"))
QUEUE_MODEL_PATH = Path(os.getenv("QUEUE_MODEL_PATH", MODEL_DIR / "xgb_queue_tuned_model.pkl"))
WAIT_MODEL_PATH = Path(os.getenv("WAIT_MODEL_PATH", MODEL_DIR / "xgb_wait_tuned_model.pkl"))
FAULT_MODEL_PATH = Path(os.getenv("FAULT_MODEL_PATH", MODEL_DIR / "lgbm_fault_tuned_model.pkl"))
ACTION_MODEL_PATH = Path(os.getenv("ACTION_MODEL_PATH", MODEL_DIR / "xgb_action_tuned_model.pkl"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", MODEL_DIR / "scaler.pkl"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", MODEL_DIR / "feature_columns.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", MODEL_DIR / "label_encoder.pkl"))

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

# Server config
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Environment
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")