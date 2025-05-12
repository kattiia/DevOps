# Stage 1: Build environment
FROM python:3.10-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir=/install/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Установка зависимостей
COPY --from=builder /install/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Копируем API и конфиг
COPY api /app/api
COPY config.yaml /app
COPY models/co2_model.h5 /app/api/models/co2_model.h5

WORKDIR /app/api
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

