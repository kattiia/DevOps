FROM python:3.10-slim
WORKDIR /app

# Устанавливаем MLflow и boto3 для работы с S3
RUN pip install --no-cache-dir mlflow boto3

EXPOSE 5000
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://terraform-s3-bucket/mlflow-artifacts
