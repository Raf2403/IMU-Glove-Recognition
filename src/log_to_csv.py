import serial
import csv
import time
import re

# --- НАСТРОЙКИ ---
COM_PORT = 'COM3' 
BAUD_RATE = 115200
OUTPUT_FILE = 'imu_data.csv'

# --- ЗАПУСК ---
label = input("Введите метку для записи (UP, DOWN, LEFT, RIGHT, IDLE): ").upper()
duration = int(input("Сколько секунд пишем? (обычно 60-120): "))

print(f"Готовься... Метка: {label}")
time.sleep(2) # Пауза чтобы взять плату в руку
print("ПОЕХАЛИ! Маши!")

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    # Открываем файл в режиме APPEND ('a'), чтобы дописывать новые жесты в конец
    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Если файл пустой, запишем заголовок (простая проверка)
        if file.tell() == 0:
             writer.writerow(['label', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                # Парсим строку вида "A:123 456 789 G:12 34 56"
                vals = re.findall(r'-?\d+', line)
                
                if len(vals) == 6:
                    # Порядок: accel_x, y, z, gyro_x, y, z
                    # Пишем: [LABEL, ax, ay, az, gx, gy, gz]
                    row = [label] + vals
                    writer.writerow(row)
                    # print(f"Saved: {row}") # Можно закомментить, чтобы не спамило
                
    print(f"\nГотово! Записано {duration} секунд для метки {label}.")

except Exception as e:
    print(f"Ошибка: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
