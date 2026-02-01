"""
NavSwap AI Model Conversion Script
Converts XGBoost and LightGBM models to ONNX format for on-device inference.

Usage:
    python convert_to_onnx.py

Prerequisites:
    pip install joblib onnxmltools skl2onnx xgboost lightgbm numpy

Input:
    - models/xgb_queue_tuned_model.pkl
    - models/xgb_wait_tuned_model.pkl
    - models/lgbm_fault_tuned_model.pkl
    - models/xgb_action_tuned_model.pkl
    - models/scaler.pkl

Output:
    - models/queue_model.onnx
    - models/wait_model.onnx
    - models/fault_model.onnx
    - models/action_model.onnx
    - models/scaler_params.json (for Dart preprocessing)
"""

import os
import json
import joblib
import numpy as np

try:
    from onnxmltools import convert_xgboost, convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
except ImportError:
    print("Installing required packages...")
    os.system("pip install onnxmltools skl2onnx")
    from onnxmltools import convert_xgboost, convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

MODELS_DIR = "models"
OUTPUT_DIR = "models"

# Expected number of input features (adjust based on your actual model)
# This should match the number of columns in X_train after preprocessing
N_FEATURES = 30  # Placeholder - update based on your actual feature count


def load_model(filename):
    """Load a pickled model."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None
    print(f"✅ Loaded: {filename}")
    return joblib.load(path)


def convert_xgb_to_onnx(model, output_name, n_features):
    """Convert XGBoost model to ONNX format."""
    if model is None:
        return False
    
    try:
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✅ Converted to: {output_name}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed for {output_name}: {e}")
        return False


def convert_lgbm_to_onnx(model, output_name, n_features):
    """Convert LightGBM model to ONNX format."""
    if model is None:
        return False
    
    try:
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type)
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✅ Converted to: {output_name}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed for {output_name}: {e}")
        return False


def export_scaler_params(scaler, output_name="scaler_params.json"):
    """Export StandardScaler parameters for Dart preprocessing."""
    if scaler is None:
        return False
    
    try:
        params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "feature_names": list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None
        }
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Exported scaler params to: {output_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to export scaler params: {e}")
        return False


def export_label_encoder(encoder, output_name="label_encoder.json"):
    """Export LabelEncoder classes for Dart decoding."""
    if encoder is None:
        return False
    
    try:
        params = {
            "classes": encoder.classes_.tolist()
        }
        
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Exported label encoder to: {output_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to export label encoder: {e}")
        return False


def main():
    print("=" * 50)
    print("NavSwap AI Model Conversion to ONNX")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load models
    print("\n📦 Loading models...")
    xgb_queue = load_model("xgb_queue_tuned_model.pkl")
    xgb_wait = load_model("xgb_wait_tuned_model.pkl")
    lgbm_fault = load_model("lgbm_fault_tuned_model.pkl")
    xgb_action = load_model("xgb_action_tuned_model.pkl")
    scaler = load_model("scaler.pkl")
    label_encoder = load_model("label_encoder.pkl")
    
    # Determine number of features from scaler if available
    n_features = N_FEATURES
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        n_features = scaler.n_features_in_
        print(f"\n📊 Detected {n_features} input features from scaler")
    
    # Convert models
    print("\n🔄 Converting models to ONNX...")
    convert_xgb_to_onnx(xgb_queue, "queue_model.onnx", n_features)
    convert_xgb_to_onnx(xgb_wait, "wait_model.onnx", n_features)
    convert_lgbm_to_onnx(lgbm_fault, "fault_model.onnx", n_features)
    convert_xgb_to_onnx(xgb_action, "action_model.onnx", n_features)
    
    # Export preprocessing params
    print("\n📤 Exporting preprocessing parameters...")
    export_scaler_params(scaler)
    export_label_encoder(label_encoder)
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("Copy the .onnx and .json files to your Flutter assets/models/ directory")
    print("=" * 50)


if __name__ == "__main__":
    main()
