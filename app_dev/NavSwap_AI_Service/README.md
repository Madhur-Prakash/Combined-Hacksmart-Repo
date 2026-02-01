# NavSwap AI Prediction Microservice

This microservice provides AI-powered predictions for the NavSwap EV Battery Management System.

## Setup

1.  **Prerequisites**:
    *   Python 3.11+
    *   Docker (optional)

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup**:
    > [!IMPORTANT]
    > You MUST place the following trained model files in the `models/` directory:
    *   `xgb_queue_tuned_model.pkl`
    *   `xgb_wait_tuned_model.pkl`
    *   `lgbm_fault_tuned_model.pkl`
    *   `xgb_action_tuned_model.pkl`
    *   `scaler.pkl`
    *   `label_encoder.pkl`

    Without these files, the service will start but predictions will fail.

4.  **Environment Variables**:
    *   `GEMINI_API_KEY`: Required for the `/explain-decision` endpoint.

## Running the Service

### Locally
```bash
uvicorn app.main:app --reload
```
Access docs at `http://localhost:8000/docs`.

### With Docker
```bash
docker build -t navswap-ai .
docker run -p 8000:8000 -v $(pwd)/models:/app/models --env-file .env navswap-ai
```

## API Endpoints

*   `POST /predict-load`: Predict queue length and wait time.
*   `POST /predict-fault`: Predict fault risk.
*   `POST /predict-action`: Recommend system action.
*   `POST /explain-decision`: Generate explanation using Gemini.
*   `GET /health`: Health check.
