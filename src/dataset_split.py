import matplotlib.pyplot as plt

# Данные
sizes = [64, 16, 20]
labels = ['Train', 'Validation', 'Test']
# Цвета как на моем графике (синий, оранжевый, зеленый)
colors = ['#38bdf8', '#fbbf24', '#4ade80'] 

plt.figure(figsize=(7, 7), dpi=120)

# Рисуем "пончик"
# wedgeprops={'width': 0.4} делает дырку (ширина кольца 0.4 от радиуса)
wedges, texts, autotexts = plt.pie(
    sizes, 
    labels=labels,
    colors=colors, 
    autopct='%1.0f%%',  # Только целые числа (64%, 16%...)
    startangle=90,      # Начинаем сверху
    pctdistance=0.80,   # Проценты ближе к краю кольца
    wedgeprops={'width': 0.4, 'edgecolor': 'w'}, # Белые границы между секторами
    textprops={'fontsize': 12}
)

# Настройка шрифтов
plt.setp(autotexts, size=12, weight="bold", color="black") # Проценты жирным
plt.setp(texts, size=14) # Подписи (Train...) обычным

plt.title('Распределение данных', fontsize=16, pad=10)
plt.tight_layout()

plt.savefig('dataset_donut.png', transparent=True)
plt.show()
