from prometheus_client import start_http_server, Gauge
import random
import time

g = Gauge('dummy_co2_metric', 'Демонстрационная метрика уровня CO2')
start_http_server(9000)  # Экспортер слушает порт 9000

if __name__ == '__main__':
    while True:
        # Устанавливаем случайное значение метрики
        g.set(random.random() * 100)
        time.sleep(5)
