"""FastAPI prediction service with Pydantic validation + Prometheus monitoring."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field
from train import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'artifacts' / 'champion.pkl'
app = FastAPI(title='Hotel Cancellation MLOps API', version='1.0.0')
model = None
log_lock = Lock()
REQUESTS = Counter('hotel_predictions_total', 'Prediction count')
LATENCY = Histogram('hotel_prediction_seconds', 'Prediction latency seconds')
LEAD_TIME = Histogram('hotel_lead_time_days', 'Lead times in live requests', buckets=(0,7,14,30,60,90,120,180,300,500))

class Reservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    no_of_adults: int = Field(ge=0, le=10)
    no_of_children: int = Field(ge=0, le=20)
    no_of_weekend_nights: int = Field(ge=0, le=30)
    no_of_week_nights: int = Field(ge=0, le=90)
    type_of_meal_plan: str
    required_car_parking_space: int = Field(ge=0, le=1)
    room_type_reserved: str
    lead_time: int = Field(ge=0, le=1000)
    arrival_year: int = Field(ge=2000, le=2100)
    arrival_month: int = Field(ge=1, le=12)
    arrival_date: int = Field(ge=1, le=31)
    market_segment_type: str
    repeated_guest: int = Field(ge=0, le=1)
    no_of_previous_cancellations: int = Field(ge=0)
    no_of_previous_bookings_not_canceled: int = Field(ge=0)
    avg_price_per_room: float = Field(ge=0)
    no_of_special_requests: int = Field(ge=0)


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail='Model is unavailable; run python train.py')
        model = joblib.load(MODEL_PATH)
    return model

@app.get('/health')
def health():
    return {'status': 'ok' if MODEL_PATH.is_file() else 'model_missing'}

@app.post('/predict')
def predict(reservation: Reservation):
    with LATENCY.time():
        record = reservation.model_dump()
        try:
            probability = float(get_model().predict_proba(pd.DataFrame([record], columns=FEATURES))[0][1])
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail='Prediction failed') from exc
    probability = max(0.0, min(1.0, probability))
    REQUESTS.inc()
    LEAD_TIME.observe(record['lead_time'])
    logfile = ROOT / 'predictions.csv'
    with log_lock:
        new = not logfile.exists() or logfile.stat().st_size == 0
        with logfile.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(['timestamp', 'lead_time', 'cancel_probability'])
            writer.writerow([datetime.now(timezone.utc).isoformat(), record['lead_time'], round(probability, 6)])
    return {'cancel_probability': round(probability, 4),
            'prediction': 'Canceled' if probability >= 0.5 else 'Not_Canceled'}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
