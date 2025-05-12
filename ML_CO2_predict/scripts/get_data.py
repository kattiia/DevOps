import os
import requests
import pandas as pd

# Создаем папку для сырых данных
os.makedirs("data/raw", exist_ok=True)

# URL с данными CO2 с Mauna Loa (NOAA)
url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
print(f"Загрузка данных с {url} ...")
response = requests.get(url)
response.raise_for_status()

# Парсим текстовые данные
lines = response.text.splitlines()
data = []
for line in lines:
    if line.startswith('#') or not line.strip():
        continue
    parts = line.split()
    year = int(parts[0])
    month = int(parts[1])
    # Средний уровень CO2 (ppm)
    co2 = float(parts[3])
    date = f"{year}-{month:02d}-01"
    data.append({"date": date, "co2": co2})

# Сохраняем как CSV
df = pd.DataFrame(data)
csv_path = "data/raw/co2_mlo.csv"
df.to_csv(csv_path, index=False)
print(f"Сырые данные сохранены в {csv_path}")