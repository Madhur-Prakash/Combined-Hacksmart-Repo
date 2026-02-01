import joblib
import numpy as np

print("Inspecting Fault Model Feature Requirements...")

try:
    fault_model = joblib.load('app/models/lgbm_fault_tuned_model.pkl')
    print(f"Fault model type: {type(fault_model)}")
    
    if hasattr(fault_model, 'feature_names_in_'):
        print(f"Fault model expects {len(fault_model.feature_names_in_)} features:")
        for i, feature in enumerate(fault_model.feature_names_in_, 1):
            print(f"  {i:2d}. {feature}")
    elif hasattr(fault_model, 'n_features_in_'):
        print(f"Fault model expects {fault_model.n_features_in_} features")
    
    # Test with dummy data
    dummy_data = np.random.randn(1, 25)
    prediction = fault_model.predict(dummy_data)
    print(f"Model works with 25 features, prediction: {prediction[0]}")
    
except Exception as e:
    print(f"Error: {e}")

print("\nCurrent preprocessing provides 23 features for queue/wait models")
print("Need to add 2 more features for fault model")