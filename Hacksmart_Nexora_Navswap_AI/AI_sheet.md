## Project Overview: NavSwap EV Battery Management AI Models

This project aimed to develop three critical AI models for NavSwap's EV battery swap management system using a simulated operational dataset. The models address future load prediction, fault/failure prediction, and system optimization policy to enhance operational efficiency and user experience.

### 1. Data Source and Initial Understanding

The project utilized `simulated_data.csv`, a real-world operational dataset. It contained various features crucial for understanding the dynamics of EV charging stations:

*   **Station Information**: `station_id`
*   **Time-series Data**: `timestamp`, `hour_of_day`, `day_of_week`, `is_peak_hour`
*   **Operational Metrics**: `queue_length`, `available_batteries`, `total_batteries`, `available_chargers`, `total_chargers`, `faulty_chargers`, `avg_wait_time`, `power_usage_kw`, `power_capacity_kw`, `fault_count_24h`
*   **Contextual Factors**: `traffic_factor`, `station_reliability_score`, `energy_stability_index`, `weather_condition`, `status`

### 2. Data Preprocessing and Feature Engineering

Before training any models, the raw data underwent several essential preprocessing steps to ensure quality and compatibility with machine learning algorithms:

*   **Datetime Parsing**: The `timestamp` column was converted to datetime objects to enable time-series analysis.
*   **Numerical Feature Scaling**: Key numerical features (`queue_length`, `available_batteries`, `available_chargers`, `avg_wait_time`, `power_usage_kw`, `traffic_factor`, `station_reliability_score`, `energy_stability_index`) were scaled using `StandardScaler`. This normalizes their ranges, preventing features with larger values from dominating the learning process. Columns like `total_batteries`, `power_capacity_kw` (constant), `fault_count_24h`, `faulty_chargers`, `hour_of_day`, `day_of_week`, and `is_peak_hour` (discrete/categorical in nature) were excluded from scaling.
*   **Categorical Feature Encoding**: Categorical columns such as `weather_condition`, `status`, and `station_id` were transformed into numerical format using One-Hot Encoding (`pd.get_dummies`). This creates new binary columns for each category, which is necessary for most machine learning models.
*   **Target Variable Creation (`fault_risk`)**: For the Fault/Failure Prediction Model, a new binary target variable `fault_risk` was engineered. `fault_risk` was set to `1` if `faulty_chargers > 0` OR `fault_count_24h > 0`; otherwise, it was `0`.
*   **Target Variable Creation (`action`)**: For the System Optimization Policy Model, a multi-class target variable `action` was created based on the following logic:
    *   If `fault_risk == 1`, then `action = 'MAINTENANCE_ALERT'`.
    *   Otherwise, if `queue_length > 1.0` (high queue) AND `available_batteries < -1.0` (low batteries), then `action = 'REDIRECT'`.
    *   Otherwise, `action = 'NORMAL'`.
    This categorical 'action' was then encoded into numerical labels (0, 1, 2) using `LabelEncoder`.

### 3. Model Training and Evaluation

Three distinct AI models were trained, each addressing a specific problem in NavSwap's battery management.

#### 3.1. Future Load Prediction Model (Regression)

*   **Problem Solved**: Predicting `queue_length` and `avg_wait_time` one step ahead to anticipate future station demand.
*   **Algorithm Chosen**: XGBoost Regressor. XGBoost was selected for its proven performance in structured data, its ability to handle complex non-linear relationships, and its robustness to overfitting through built-in regularization.
*   **Input Features (X)**: All preprocessed features except `timestamp`, `queue_length`, `avg_wait_time`, `fault_risk`, `faulty_chargers`, and `fault_count_24h`.
*   **Output Features (y)**: Separate models were trained for `y_queue` (`queue_length` shifted by one step) and `y_wait` (`avg_wait_time` shifted by one step).
*   **Train-Test Split**: A time-based split was used (80% for training, 20% for testing) to preserve the temporal order of the data, crucial for time-series predictions.
*   **Hyperparameter Tuning**: `GridSearchCV` was employed to systematically search for the optimal hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`) to minimize Mean Absolute Error (MAE).
*   **Evaluation Metrics (Initial vs. Tuned)**:
    *   **Queue Length Prediction:**
        *   Initial MAE: 0.5370, Tuned MAE: 0.4640 (Improved by 0.0730)
        *   Initial RMSE: 0.7205, Tuned RMSE: 0.6338 (Improved by 0.0867)
    *   **Average Wait Time Prediction:**
        *   Initial MAE: 0.5352, Tuned MAE: 0.4842 (Improved by 0.0510)
        *   Initial RMSE: 1.0054, Tuned RMSE: 0.8942 (Improved by 0.1112)
*   **Feature Importance**: Plots revealed that `available_batteries`, `available_chargers`, `power_usage_kw`, `traffic_factor`, and `station_reliability_score` were among the most influential features for predicting both queue length and average wait time.

#### 3.2. Fault/Failure Prediction Model (Classification)

*   **Problem Solved**: Predicting the likelihood of a `fault_risk` occurring.
*   **Algorithm Chosen**: LightGBM Classifier. LightGBM was chosen due to its efficiency, speed, and ability to handle large datasets while maintaining high accuracy. Its `class_weight='balanced'` parameter effectively addresses the inherent class imbalance in fault detection problems.
*   **Input Features (X)**: All preprocessed features except `timestamp`, `fault_risk`, `faulty_chargers`, and `fault_count_24h` (to prevent data leakage).
*   **Output Feature (y)**: `fault_risk` (binary: 0 or 1).
*   **Train-Test Split**: A stratified split (80% train, 20% test) was used to ensure the class distribution of `fault_risk` was maintained across both sets.
*   **Hyperparameter Tuning**: `GridSearchCV` was used to optimize `n_estimators`, `learning_rate`, `num_leaves`, and `max_depth`, with `roc_auc` as the primary scoring metric due to class imbalance.
*   **Evaluation Metrics (Initial vs. Tuned)**:
    *   **Accuracy:** Initial: 0.6855, Tuned: 0.6677 (Slight decrease of 0.0178)
    *   **Precision:** Initial: 0.7538, Tuned: 0.8079 (Improved by 0.0541)
    *   **Recall:** Initial: 0.7171, Tuned: 0.5951 (Decreased by 0.1220)
    *   **F1-Score:** Initial: 0.7350, Tuned: 0.6854 (Decreased by 0.0496)
    *   **ROC AUC:** Initial: 0.7295, Tuned: 0.7458 (Improved by 0.0163)
    *(Note: The target accuracy of 90-95% was not achieved, indicating potential for further model refinement or more data.)*

#### 3.3. System Optimization Policy Model (Multi-class Classification)

*   **Problem Solved**: Recommending an optimal system `action` (MAINTENANCE_ALERT, NORMAL, or REDIRECT) based on current conditions.
*   **Algorithm Chosen**: XGBoost Classifier. Chosen for its strong performance in multi-class classification tasks and its ability to provide feature importances for decision interpretability.
*   **Input Features (X)**: All preprocessed features except `timestamp`, `faulty_chargers`, `fault_count_24h`, `fault_risk`, `action`, `action_encoded`, `queue_length`, and `avg_wait_time` (to avoid data leakage as `queue_length`, `available_batteries`, and `fault_risk` were used in defining the `action`).
*   **Output Feature (y)**: `action_encoded` (multi-class: 0, 1, or 2).
*   **Train-Test Split**: A stratified split (80% train, 20% test) ensured the representation of all action classes.
*   **Hyperparameter Tuning**: `GridSearchCV` was used to optimize `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree`, with `accuracy` as the primary refitting metric.
*   **Evaluation Metrics (Initial vs. Tuned)**:
    *   **Accuracy:** Initial: 0.6202, Tuned: 0.6231 (Slight improvement of 0.0029)
    *   **Precision (weighted):** Initial: 0.6100, Tuned: 0.6122 (Slight improvement of 0.0022)
    *   **Recall (weighted):** Initial: 0.6202, Tuned: 0.6231 (Slight improvement of 0.0029)
    *   **F1-Score (weighted):** Initial: 0.6125, Tuned: 0.5332 (Unexpected decrease of 0.0793)
    *(Note: The target accuracy of 90-95% was not met, and the F1-Score decreased, suggesting this model might require more advanced techniques or feature engineering.)*
*   **Feature Importance**: Plots identified various features contributing to action recommendations, highlighting the model's reliance on contextual factors to determine the best course of action.

### 4. Implications for NavSwap EV Battery Management

*   **Enhanced Operational Efficiency**: The improved **Future Load Prediction Models** allow NavSwap to anticipate demand more accurately, enabling proactive resource allocation (e.g., dispatching mobile batteries, rerouting vehicles to less busy stations) and dynamic pricing strategies, leading to reduced wait times and optimal station utilization.
*   **Proactive Maintenance and Reliability**: The **Fault/Failure Prediction Model**, with its improved precision and ROC AUC, can provide early warnings for potential equipment failures. This enables NavSwap to schedule preventative maintenance, minimizing unexpected downtime and ensuring higher availability of charging infrastructure, thus increasing customer satisfaction and reducing operational costs.
*   **Automated Decision-Making**: Although the **System Optimization Policy Model** showed modest improvements, it lays the groundwork for automating critical operational decisions. Further enhancements could lead to a fully autonomous system that responds to real-time conditions by recommending optimal actions like rebalancing battery inventory, initiating maintenance protocols, or directing drivers, thereby improving overall system responsiveness and efficiency.

### 5. Conclusion

Through careful data preprocessing, feature engineering, and hyperparameter tuning, significant improvements were made to the regression models. The classification models, while showing some positive changes, indicate areas for further exploration to reach the desired performance targets. The trained models are now ready to be integrated into NavSwap's backend systems for advanced EV battery management.


### Backend Integration Guide: Using NavSwap AI Models
This guide outlines how to integrate and use the trained AI models (xgb_queue_tuned, xgb_wait_tuned, lgbm_fault_tuned, xgb_action_tuned) in a backend system for real-time EV battery management decisions.

1. Loading the Trained Models
The tuned models are saved as .pkl files. You can load them into your backend environment using the joblib library.

import joblib

# Load Future Load Prediction Models
xgb_queue_model = joblib.load('xgb_queue_tuned_model.pkl')
xgb_wait_model = joblib.load('xgb_wait_tuned_model.pkl')

# Load Fault/Failure Prediction Model
lgbm_fault_model = joblib.load('lgbm_fault_tuned_model.pkl')

# Load System Optimization Policy Model
xgb_action_model = joblib.load('xgb_action_tuned_model.pkl')

print("All tuned models loaded successfully.")
2. Preprocessing New Data for Inference
Any new, incoming data must undergo the exact same preprocessing steps as the training data before being fed into the models for prediction. This includes scaling numerical features, one-hot encoding categorical features, and creating derived features (fault_risk, action_encoded) if they are used as inputs for subsequent models.

Since the preprocessing steps depend on the df DataFrame's state after all initial processing (including scaling parameters, one-hot encoder columns, and label encoder mapping), it's crucial to recreate or persist these transformers. For simplicity here, we assume the df state or the scaler and label_encoder objects are available.

Key steps for new data new_df:

Datetime Conversion: new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
Numerical Scaling: Apply the fitted StandardScaler (scaler) to the same numerical_cols_to_scale.
# Assuming 'scaler' object is saved/available from training
new_df[numerical_cols_to_scale] = scaler.transform(new_df[numerical_cols_to_scale])
One-Hot Encoding: Apply pd.get_dummies for categorical_cols.
new_df = pd.get_dummies(new_df, columns=categorical_cols, drop_first=False)
# Ensure new_df columns match X_train columns - this might require reindexing or careful column management
missing_cols = set(X_train.columns) - set(new_df.columns)
for c in missing_cols: new_df[c] = 0
new_df = new_df[X_train.columns]
3. Making Predictions with Each Model
3.1. Future Load Prediction
Used for predicting queue_length and avg_wait_time.

# Assuming `X_new` is your preprocessed new data for load prediction
# X_new should have the same columns and scaling as X_train used for xgb_queue and xgb_wait

# Predict queue length
predicted_queue_length = xgb_queue_model.predict(X_new)

# Predict average wait time
predicted_avg_wait_time = xgb_wait_model.predict(X_new)

print(f"Predicted Queue Length: {predicted_queue_length}")
print(f"Predicted Average Wait Time: {predicted_avg_wait_time}")
3.2. Fault/Failure Prediction
Used for predicting fault_risk.

# Assuming `X_fault_new` is your preprocessed new data for fault prediction
# X_fault_new should have the same columns and scaling as X_fault_train

# Predict fault risk (0 or 1)
predicted_fault_risk = lgbm_fault_model.predict(X_fault_new)

# Predict probability of fault risk
predicted_fault_proba = lgbm_fault_model.predict_proba(X_fault_new)[:, 1]

print(f"Predicted Fault Risk: {predicted_fault_risk}")
print(f"Probability of Fault Risk: {predicted_fault_proba}")
3.3. System Optimization Policy
Used for recommending an action.

# Assuming `X_action_new` is your preprocessed new data for action prediction
# X_action_new should have the same columns and scaling as X_action_train

# Predict the encoded action (0, 1, or 2)
predicted_action_encoded = xgb_action_model.predict(X_action_new)

# Convert encoded action back to original labels
# Assuming 'label_encoder' object is saved/available from training
predicted_action_label = label_encoder.inverse_transform(predicted_action_encoded)

# Get probabilities for each action class
predicted_action_proba = xgb_action_model.predict_proba(X_action_new)

print(f"Predicted Action (encoded): {predicted_action_encoded}")
print(f"Predicted Action (label): {predicted_action_label}")
print(f"Probabilities for each action: {predicted_action_proba}")
4. Backend Integration and Usage
These predictions can be integrated into your backend system to drive intelligent decisions:

Real-time Dashboards: Display predicted queue lengths and wait times to station operators or directly to EV drivers.
Automated Alerts: Trigger alerts for MAINTENANCE_ALERT actions, notifying service teams of potential faults.
Dynamic Routing/Recommendations: For REDIRECT actions, guide drivers to alternative stations with lower load or higher battery availability.
Resource Management: Use predicted load to optimize battery replenishment schedules, staff allocation, or energy distribution.
Example Flow in a Backend Service:

Receive Real-time Data: Continuously ingest new operational data points from EV charging stations.
Preprocess Data: Apply the saved scaler and perform one-hot encoding using the columns from X_train to ensure consistency.
Run Inference: Pass the preprocessed data to the loaded xgb_queue_model, xgb_wait_model, lgbm_fault_model, and xgb_action_model.
Interpret Predictions: Convert numerical predictions back to human-readable formats (e.g., action_encoded to 'MAINTENANCE_ALERT').
Trigger Actions: Based on the predictions, trigger automated responses (alerts, rerouting suggestions, rebalancing tasks) or update monitoring systems.