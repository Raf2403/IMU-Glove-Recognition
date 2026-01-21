import tensorflow as tf
from tensorflow.keras.utils import plot_model

# 1. Загружаем твою обученную модель
# (убедись, что gesture_model.h5 лежит рядом)
model = tf.keras.models.load_model('gesture_model.h5')

# 2. Рисуем горизонтально (LR = Left to Right)
plot_model(model, 
           to_file='model_horizontal.png', 
           show_shapes=True,        # Показать размеры (Input: 50x6)
           show_layer_names=True,   # Показать имена слоев
           rankdir='LR')            # Горизонтальная ориентация

print("Готово! Открывай файл model_horizontal.png")
