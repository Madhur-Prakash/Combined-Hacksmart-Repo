# NavSwap AI Prediction Microservice

## What This Service Does

This is the **AI Analytics Layer** for NavSwap - a Smart EV Battery Swap Management System. The service provides intelligent predictions and decisions to optimize EV battery swap operations.

## Core Functions

### 1. **Load Prediction** (`/predict-load`)
- Predicts how many vehicles will be in queue
- Estimates wait times for customers
- Uses XGBoost models trained on historical data

### 2. **Fault Prediction** (`/predict-fault`) 
- Detects potential system failures before they happen
- Calculates fault probability (0-100%)
- Classifies risk as LOW/MEDIUM/HIGH
- Uses LightGBM model for reliability analysis

### 3. **Action Recommendation** (`/predict-action`)
- Decides optimal system response:
  - **NORMAL**: Continue regular operations
  - **REDIRECT**: Send customers to other stations
  - **MAINTENANCE_ALERT**: Schedule immediate maintenance
- Provides confidence scores for each action

### 4. **AI Explanation** (`/explain-decision`)
- Uses Google Gemini 2.5 Flash AI to explain decisions
- Converts technical predictions into human-readable explanations
- Helps operators understand why AI made specific recommendations

## Input Data Required

Each prediction needs:
- Station ID and timestamp
- Current queue length
- Battery levels and energy demand
- Weather conditions
- Station reliability metrics
- Energy grid stability

## How It Works

1. **Data Preprocessing**: Normalizes input data using trained scalers
2. **Model Inference**: Runs predictions through pre-trained ML models
3. **Decision Logic**: Combines predictions to recommend actions
4. **Explanation**: Uses Gemini AI to explain the reasoning

## Integration

This service sits between:
- **Input**: IoT sensors from EV stations
- **Output**: Recommendation engine and admin dashboard

## Technology Stack

- **FastAPI**: High-performance async web framework
- **XGBoost/LightGBM**: Machine learning models
- **Google Gemini 2.5 Flash**: Explainable AI
- **Docker**: Containerized deployment