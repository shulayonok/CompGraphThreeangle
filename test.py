import numpy as np
import matplotlib.pyplot as plt
import math as m

# Размер
n = 256

# Создаём numpy массив размерности (N,N,3)
base_colour = 139  # DarkRed
img = np.zeros((n, n, 3), dtype=np.uint8)  # Изначально фон чёрный
for i in range(n):                         # Изменим цвет фона
    for j in range(n):
        for k in range(3):
            img[i][j][k] = 255

# Алгоритм Брезенхема
def Bresenham(x0, y0, x1, y1, N, image, color):
    dx = x1 - x0
    dy = y1 - y0
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
    if abs(dx) > abs(dy):
        sx, sy, m_abs, b_abs = sign_x, 0, abs(dy), abs(dx)
    else:
        sx, sy, m_abs, b_abs = 0, sign_y, abs(dx), abs(dy)
    error = b_abs / 2
    image[N - 1 - int(y0), int(x0)] = [(color * (1 - m.sqrt((N / 2 - x0) ** 2 + (N / 2 - x0) ** 2) / N)), 0, 0]
    for i in range(b_abs):
        error -= m_abs
        if error < 0:
            error += b_abs
            x0 += sign_x
            y0 += sign_y
        else:
            x0 += sx
            y0 += sy
        image[N - 1 - int(y0), int(x0)] = [(color * (1 - m.sqrt((N / 2 - x0) ** 2 + (N / 2 - x0) ** 2) / N)), 0, 0]

pt0 = [50, 50]
pt1 = [50, 150]

versh = np.array([pt0, pt1], dtype = np.float32).T
Bresenham(versh[0][0], versh[0][1], versh[1][0], versh[1][1], n, img, base_colour)

# Отображение и сохранение в файл
plt.figure()
plt.imshow(img)
plt.show()