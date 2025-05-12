import os
import yaml
import mlflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import boto3
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

# Читаем конфигурацию
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Настройка MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Загружаем данные
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

seq_len = config["model"]["sequence_length"]
hidden_units = config["model"]["hidden_units"]
dropout = config["model"]["dropout"]
epochs = config["model"]["epochs"]
batch_size = config["model"]["batch_size"]
learning_rate = config["model"]["learning_rate"]

# Создаем модель LSTM
model = Sequential([
    LSTM(hidden_units, input_shape=(seq_len, 1)),
    Dropout(dropout),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучаем модель
with mlflow.start_run():
    # Логируем параметры
    mlflow.log_param("hidden_units", hidden_units)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=2
    )

    # Оцениваем модель и логируем метрики
    loss = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("mse", loss)
    mlflow.log_metric("rmse", np.sqrt(loss))
    print(f"Test MSE: {loss:.4f}, RMSE: {np.sqrt(loss):.4f}")

    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    # Устанавливаем session клиент явно (MLflow его подтянет через boto3 автоматически)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("AWS_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")

    # Сохраняем модель локально и в MLflow
    os.makedirs("models", exist_ok=True)
    model_path = "models/co2_model.h5"
    model.save(model_path)
    mlflow.keras.log_model(model, artifact_path="co2_model")