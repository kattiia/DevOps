FROM python:3.10-slim
WORKDIR /exporter

# Устанавливаем Prometheus client
RUN pip install --no-cache-dir prometheus_client

COPY monitoring/exporter.py /exporter/exporter.py
EXPOSE 9000
CMD ["python", "exporter.py"]
