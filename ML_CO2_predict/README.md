# 🌿 CO₂ Time Series Prediction

Проект по предсказанию уровня CO₂ на основе временного ряда с использованием LSTM-модели и полной инфраструктурой MLOps.

---

## 📁 Структура проекта
├── .dvc/config # Конфигурация dvc

├── api/ # FastAPI-приложение для инференса

├── data/

│ ├── raw/ # Исходные данные (скачиваются)

│ └── processed/ # Обработанные NumPy-массивы

├── mlflow/

│ └── Dockerfile # Локальный MLflow Tracking сервер

├── models/

│ ├── .gitignore # Игнорировать модель в Git

│ └── co2_model.h5 # LSTM-модель (локальная резервная копия)

├── monitoring/

│ ├── Dockerfile # Экспортёр метрик

│ └── exporter.py # Отправка метрик в Prometheus

├── scripts/

│ ├── dvc_pipeline.sh # Скрипт запуска dvc

│ ├── get_data.py # Скачивание данных NOAA

│ └── prepare_data.py # Подготовка npy-файлов

└── trainer/

├── Dockerfile # Обучение модели

├── train.py # Скрипт обучения и логирования в MLflow

├── .env # Переменные окружения

├── .gitlab-ci.yml # CI/CD pipeline

├── config.yaml # Конфигурация модели и путей

├── docker-compose.yml # Сервисы проекта

├── dvc.yaml # DVC-пайплайн


├── environment.yml # YAML для Conda (опц.)

├── params.yaml # DVC параметры

├── prometheus.yml # Конфиг Prometheus

├── README.md # Документация

└── requirements.txt # pip-зависимости

---
## 🛠️ Используемые технологии и инструменты

| Инструмент        | Назначение                                                                 |
|-------------------|---------------------------------------------------------------------------|
| **DVC**           | Версионирование данных и моделей, автоматизация пайплайнов                |
| **MLflow**        | Логирование параметров и метрик, регистрация модели                       |
| **S3 (Yandex)**   | Хранение сырого и обработанного датасета, артефактов моделей              |
| **Docker**        | Изолированная среда для всех сервисов                                     |
| **FastAPI**       | REST API для предсказаний                                                  |
| **Prometheus**    | Сбор метрик (в том числе скорости и ошибок инференса)                     |
| **Grafana**       | Визуализация мониторинга                                                   |
| **CI/CD (GitLab)**| Автоматическая сборка, обучение, пуш модели и деплой                      |

---

## 🔁 Как работают компоненты

- **DVC** автоматически скачивает данные, запускает обработку, обучение и пушит артефакты.
- **MLflow** отслеживает параметры, логирует метрики, сохраняет модель.
- **FastAPI** поднимает сервис и при старте загружает модель из S3.
- **Prometheus Exporter** в `monitoring/exporter.py` отправляет кастомные метрики в Prometheus.
- **Grafana** подключена к Prometheus и позволяет визуализировать, например, latency/загрузку API.

---

## 🚀 Запуск проекта

1. **Создайте `.env` файл**:
AWS_ACCESS_KEY_ID=...

AWS_SECRET_ACCESS_KEY=...

S3_BUCKET=...

MLFLOW_TRACKING_URI=http://localhost:5000

2. Заполните config.yaml, .dcv/config и params.yaml

3. **Запуск всех сервисов**:

```bash
docker-compose up --build

```

4. **Мониторинг**:

Prometheus: http://localhost:9090

Grafana: http://localhost:3000 (логин: admin, пароль: admin)

MLflow: http://localhost:5000

FastAPI: http://localhost:8000/docs

## DVC пайплайн
get_data     -> prepare_data  (npy, config.yaml) -> train (model.h5)

Запуск вручную:
```bash
bash scripts/dvc_pipeline.sh
```