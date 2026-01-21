import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных (путь к CSV из датасета)
df = pd.read_csv('imu_data.csv')  # Проверь имя файла в датасете

features = df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
labels = df['label'].values 

# 3. Энкодинг меток (жесты в цифры)
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# 4. Нормализация (акселерометр и гироскоп в разных масштабах)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. Разбивка на окна (например, 50 сэмплов = 1 жест)
WINDOW_SIZE = 50
STEP = 25  # Перекрытие окон

def create_windows(data, labels, window_size, step):
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size-1])  # метка последнего сэмпла
    return np.array(X), np.array(y)

X, y = create_windows(features_scaled, labels_encoded, WINDOW_SIZE, STEP)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Модель (простая 1D-CNN для TinyML)
num_classes = len(encoder.classes_)

model = keras.Sequential([
    keras.layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 6)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 8. Обучение
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 9. Оценка
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# 10. Сохранение
model.save('gesture_model.h5')

# 11. Конвертация в TFLite (для STM32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Квантизация (меньше размер)
tflite_model = converter.convert()

with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель сохранена: gesture_model.tflite")
print(f"Размер: {len(tflite_model) / 1024:.1f} KB")

# Данные из истории обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Создаем график
plt.figure(figsize=(12, 5))

# График точности (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# График потерь (Loss)
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() # Покажет окно с графиком
# plt.savefig('training_history.png') # Если хочешь сохранить в файл
