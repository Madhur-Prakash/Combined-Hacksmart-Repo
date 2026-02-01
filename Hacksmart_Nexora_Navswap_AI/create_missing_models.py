import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Create StandardScaler (dummy - replace with your training scaler)
scaler = StandardScaler()
# Fit with dummy data matching your features
dummy_data = np.random.randn(100, 8)  # 8 features
scaler.fit(dummy_data)

# Create feature columns list (update with your actual features)
feature_columns = [
    'current_queue', 
    'battery_level', 
    'energy_demand', 
    'weather_temp',
    'is_weekend', 
    'hour_of_day', 
    'station_reliability', 
    'energy_stability'
]

# Create LabelEncoder for actions
label_encoder = LabelEncoder()
actions = ['NORMAL', 'REDIRECT', 'MAINTENANCE_ALERT']
label_encoder.fit(actions)

# Save all preprocessing objects
joblib.dump(scaler, 'app/models/scaler.pkl')
joblib.dump(feature_columns, 'app/models/feature_columns.pkl')
joblib.dump(label_encoder, 'app/models/label_encoder.pkl')

print("✅ Created missing preprocessing files:")
print("- scaler.pkl")
print("- feature_columns.pkl") 
print("- label_encoder.pkl")
print("\n⚠️  Replace scaler.pkl with your actual training scaler for best results")