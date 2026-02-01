import joblib
import numpy as np

print("🔍 Inspecting Model Feature Requirements...")

# Load the actual feature columns from your model
try:
    feature_columns = joblib.load('app/models/feature_columns.pkl')
    print(f"Feature columns loaded: {len(feature_columns)} features")
    print("Features expected by model:")
    for i, feature in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {feature}")
except Exception as e:
    print(f"Error loading feature columns: {e}")

# Load and inspect one of your models
try:
    model = joblib.load('app/models/xgb_queue_tuned_model.pkl')
    print(f"\nModel type: {type(model)}")
    
    # Try to get feature info from XGBoost model
    if hasattr(model, 'feature_names_in_'):
        print(f"Model expects {len(model.feature_names_in_)} features:")
        for i, feature in enumerate(model.feature_names_in_, 1):
            print(f"  {i:2d}. {feature}")
    elif hasattr(model, 'n_features_in_'):
        print(f"Model expects {model.n_features_in_} features")
    
except Exception as e:
    print(f"Error loading model: {e}")

# Test with dummy data to see expected shape
try:
    # Create dummy data with 23 features (what model expects)
    dummy_data = np.random.randn(1, 23)
    prediction = model.predict(dummy_data)
    print(f"\n✅ Model works with 23 features, prediction: {prediction[0]}")
except Exception as e:
    print(f"Error with 23 features: {e}")

print("\n📋 Current API features (8):")
current_features = [
    'current_queue', 'battery_level', 'energy_demand', 'weather_temp',
    'is_weekend', 'hour_of_day', 'station_reliability', 'energy_stability'
]
for i, feature in enumerate(current_features, 1):
    print(f"  {i}. {feature}")