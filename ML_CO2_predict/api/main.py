from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import boto3
from starlette.responses import Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

class PredictRequest(BaseModel):
    values: list

app = FastAPI(title="CO2 Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Счетчик Prometheus для количества запросов предсказаний
PREDICT_REQUESTS = Counter('predict_requests_total', 'Количество запросов на /predict')

MODEL_PATH = "models/co2_model.h5"
bucket = os.getenv("AWS_BUCKET_NAME", "##############")
model_key = os.getenv("MODEL_S3_KEY", "models/co2_model.h5")

# Загружаем модель из S3 при старте, если она не локально
if not os.path.exists(MODEL_PATH):
    print("Загрузка модели из S3...")
    s3 = boto3.client('s3', endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL"))
    s3.download_file(bucket, model_key, MODEL_PATH)

from tensorflow.keras.models import load_model
MODEL = load_model(MODEL_PATH)
print("Модель загружена.")

@app.post("/predict")
def predict(request: PredictRequest):
    PREDICT_REQUESTS.inc()
    values = np.array(request.values, dtype=float)
    values = values.reshape((1, -1, 1))
    pred = MODEL.predict(values)
    prediction = float(pred[0][0])
    return {"prediction": prediction}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
