import os
import numpy as np
import pandas as pd
import yaml

# Читаем конфигурацию
with open("config.yaml") as f:
    config = yaml.safe_load(f)
seq_len = config["model"]["sequence_length"]

# Загружаем сырые данные
df = pd.read_csv("data/raw/co2_mlo.csv")
values = df["co2"].values

# Подготавливаем последовательности данных
X, y = [], []
for i in range(len(values) - seq_len):
    X.append(values[i : i + seq_len])
    y.append(values[i + seq_len])
X = np.array(X)
y = np.array(y)

# Добавляем размерность для LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Разбиваем на обучающую и тестовую выборки (последний 20% для теста)
test_size = 0.2
split_idx = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Сохраняем подготовленные данные
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
print("Данные подготовлены и сохранены в data/processed")