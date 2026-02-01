# NavSwap AI Prediction Microservice

Production-ready AI microservice for NavSwap Smart EV Battery Swap Management System with comprehensive operational intelligence.

for detailed references , in the repository please refer AI_sheet.md , INSTALLATION guide.md and what is being done.md file 

## Architecture

This service represents the AI Analytics Layer and handles:
- **Future Load Prediction** (queue length, wait time)
- **Fault/Failure Prediction** (system reliability)
- **System Optimization Policy Decision** (NORMAL/REDIRECT/MAINTENANCE_ALERT)
- **Traffic Analysis** (micro-traffic, congestion forecasting)
- **Battery Logistics** (rebalancing, inventory management)
- **Staff Operations** (resource allocation, skill matching)
- **Demand Forecasting** (customer arrivals, battery demand)
- **Partner Coordination** (storage network optimization)
- **Explainable AI Layer** using Google Gemini 2.5 Flash

## AI Models (8 Total)

### Core Prediction Models (4)
1. **XGBoost Queue Model** - Predicts vehicle queue length
2. **XGBoost Wait Time Model** - Estimates customer wait times
3. **LightGBM Fault Model** - Detects system reliability issues
4. **XGBoost Action Model** - Recommends operational actions

### Operations Optimization Models (4)
5. **Traffic Forecast Model** - Future traffic pattern analysis
6. **Micro Traffic Model** - Real-time congestion monitoring
7. **Battery Rebalance Model** - Logistics optimization
8. **Stock Order Model** - Inventory management
9. **Staff Diversion Model** - Human resource allocation
10. **Tieup Storage Model** - Partner network coordination
11. **Customer Arrival Model** - Demand forecasting
12. **Battery Demand Model** - Supply planning

### AI Enhancement Layer
- **Station Recommender** - Intelligent station ranking
- **Gemini 2.5 Flash LLM** - Explainable AI insights

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your Gemini API key:
```bash
cp .env.example .env
# Edit .env file with your actual API key
```

### 3. Add Model Files
Place your trained models in `app/models/`:
- Core models: `xgb_queue_tuned_model.pkl`, `xgb_wait_tuned_model.pkl`, etc.
- Operations models: `traffic_forecast_model.pkl`, `battery_rebalance_model.pkl`, etc.
- Preprocessing: `scaler.pkl`, `feature_columns.pkl`, `label_encoder.pkl`

### 4. Run the Service
```bash
python -m app.main
```

## API Endpoints

### Individual Model Endpoints
- `POST /api/v1/predict-load` - Queue & wait time prediction
- `POST /api/v1/predict-fault` - System reliability analysis
- `POST /api/v1/predict-action` - Operational decision making
- `POST /api/v1/explain-decision` - AI explanation for decisions

### Unified Intelligence Endpoints
- `POST /api/v1/smart-recommend` - Station recommendation with 4 core models
- `POST /api/v1/smart-operations` - Complete operational optimization (all 8+ models)

### Health & Monitoring
- `GET /health` - Service health check
- `GET /` - Service status
- `GET /docs` - Interactive API documentation

## Docker Deployment

```bash
docker build -t navswap-ai-service .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" navswap-ai-service
```

## Integration

Service integrates with:
- **IoT Sensors** (input data)
- **Admin Dashboard** (operational insights)
- **User Mobile App** (station recommendations)
- **Partner Networks** (storage coordination)

Service runs on `http://localhost:8000`
