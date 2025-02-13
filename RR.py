import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Генерация синтетических данных
N = 200
x = np.linspace(-10, 10, N)
y_true = np.sin(x) + 0.1 * x ** 2
noise = np.random.normal(0, 0.5, N)
y = y_true + noise

# Решейп x для подачи в модель (нужно сделать из вектора матрицу с одним столбцом)
x = x.reshape(-1, 1)

# 2. Разделение данных на обучающую и тестовую выборки
train_size = int(0.9 * N)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Создание модели нейронной сети
model = keras.Sequential([
    keras.layers.Dense(50, activation='relu', input_shape=(1,)),  # Скрытый слой
    keras.layers.Dense(1)  # Выходной слой
])

# 4. Компиляция модели
model.compile(optimizer='adam', loss='mse')

# 5. Обучение модели
history = model.fit(x_train, y_train, epochs=200, verbose=0)

# 6. Предсказания модели
y_pred = model.predict(x)

# 7. Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label='Истинная функция', color='blue')
plt.scatter(x_train, y_train, label='Обучающие данные', color='green', alpha=0.5)
plt.scatter(x_test, y_test, label='Тестовые данные', color='orange', alpha=0.5)
plt.plot(x, y_pred, label='Предсказания нейросети', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Нелинейная регрессия с помощью нейронной сети')
plt.show()

# График функции потерь
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='MSE')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.title('График функции потерь (MSE) в процессе обучения')
plt.show()
