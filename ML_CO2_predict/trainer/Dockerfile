# Stage 1: Build environment
FROM python:3.10-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir=/install/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Установка зависимостей из wheel
COPY --from=builder /install/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Копируем всё остальное
COPY . /app

# Установка git (нужен DVC)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Инициализируем пустой git (нужно DVC)
RUN git init && git config user.email "123@example.com" && git config user.name "Name"

RUN sh -c '[ -d ".dvc" ] || (echo "📦 Инициализация DVC..." && dvc init --no-scm)'
RUN sh -c '! dvc remote list | grep -q "^storage" && \
  echo "🔗 Добавление remote-хранилища..." && \
  dvc remote add -d storage s3://${S3_BUCKET}/co2-project || \
  echo "✅ Remote 'storage' уже существует, пропускаем добавление"'
RUN dvc config core.remote storage

CMD ["bash", "scripts/dvc_pipeline.sh"]

