import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import math as m
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def Bresenham(x0, y0, x1, y1, N, image, color):
    dx = x1 - x0
    dy = y1 - y0
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
    if abs(dx) > abs(dy):
        sx, sy, m_abs, b_abs = sign_x, 0, abs(dy), abs(dx)
    else:
        sx, sy, m_abs, b_abs = 0, sign_y, abs(dx), abs(dy)
    error, t = b_abs / 2, 0
    image[N - 1 - int(y0), int(x0)] = [(color * (1 - m.sqrt((N / 2 - x0) ** 2 + (N / 2 - x0) ** 2) / N)), 0, 0]
    while t < b_abs:
        error -= m_abs
        if error < 0:
            error += b_abs
            x0 += sign_x
            y0 += sign_y
        else:
            x0 += sx
            y0 += sy
        t += 1
        image[N - 1 - int(y0), int(x0)] = [(color * (1 - m.sqrt((N / 2 - x0) ** 2 + (N / 2 - x0) ** 2) / N)), 0, 0]

def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr

def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr

def diagMatr(s):  # Масштабирование чётное
    mtr = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    return mtr

def to_proj_coords(x):
    r, c = x.shape
    x = np.concatenate([x, np.ones((1, c))], axis=0)
    return x

def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x

def create_rectangle():
    a = [10, 10]
    b = [500, 10]
    c = [500, 500]
    d = [10, 500]
    x_1 = np.array([a, b], dtype=np.float32).T
    x_2 = np.array([b, c], dtype=np.float32).T
    x_3 = np.array([c, d], dtype=np.float32).T
    x_4 = np.array([d, a], dtype=np.float32).T
    x = np.array([x_1, x_2, x_3, x_4])
    return x

def create_threeangle():
    while True:
        x0 = round(random.uniform(30, 470), 3)
        y0 = round(random.uniform(30, 470), 3)
        length = 0
        while length != 50:
            x1 = round(random.uniform(30, 470), 3)
            y1 = round(random.uniform(30, 470), 3)
            length = round(m.sqrt((x1-x0)**2 + (y1-y0)**2), 3)
        x2 = round((x1 - x0) * m.cos(m.pi/3) - (y1 - y0) * m.sin(m.pi/3) + x0, 3)
        y2 = round((x1 - x0) * m.sin(m.pi/3) + (y1 - y0) * m.cos(m.pi/3) + y0, 3)
        if 10 < x2 < 500 and 10 < y2 < 500:
            break
    a = [x0, y0]
    b = [x1, y1]
    c = [x2, y2]
    cen = np.array([(x0+x1+x2)/3, (y0+y1+y2)/3])
    x_1 = np.array([a, b], dtype=np.float32).T
    x_2 = np.array([b, c], dtype=np.float32).T
    x_3 = np.array([c, a], dtype=np.float32).T
    return np.array([x_1, x_2, x_3]), cen

def create_space():
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, :3] = 255
    create_rectangle()
    return image

def touch(a):
    for i in range(len(a)):
        if a[i][0][0] >= 500:
            return 1
        elif a[i][0][0] <= 10:
            return 2
        elif a[i][1][0] >= 500:
            return 3
        elif a[i][1][0] <= 10:
            return 4


N = 200  # frames count
size = 510
base_colour = 139  # DarkRed

y, center = create_threeangle()

y_proj0 = to_proj_coords(y[0])
y_proj1 = to_proj_coords(y[1])
y_proj2 = to_proj_coords(y[2])

fig = plt.figure()
frames = []
v_x = 0
v_y = 0
sign_x = 1
sign_y = 1
z = 5
S = 1

for i in range(N):
    ang = -i * 2 * np.pi / N
    T = shiftMatr(-center)
    R = rotMatr(ang)
    v_x += sign_x * z
    v_y += sign_y * z
    S += 0.01
    y_new0 = shiftMatr([v_x, v_y]) @ np.linalg.inv(T) @ diagMatr(S) @ T @ np.linalg.inv(T) @ R @ T @ y_proj0
    y_new1 = shiftMatr([v_x, v_y]) @ np.linalg.inv(T) @ diagMatr(S) @ T @ np.linalg.inv(T) @ R @ T @ y_proj1
    y_new2 = shiftMatr([v_x, v_y]) @ np.linalg.inv(T) @ diagMatr(S) @ T @ np.linalg.inv(T) @ R @ T @ y_proj2
    y_new = np.array([y_new0, y_new1, y_new2])
    if touch(y_new) == 1:
        sign_x = -1
    elif touch(y_new) == 2:
        sign_x = 1
    elif touch(y_new) == 3:
        sign_y = -1
    elif touch(y_new) == 4:
        sign_y = 1

    img = create_space()
    x = create_rectangle()
    for j in range(len(x)):
        Bresenham(x[j][0][0], x[j][0][1], x[j][1][0], x[j][1][1], size, img, base_colour)
    for k in range(len(y)):
        Bresenham(y_new[k][0][0], y_new[k][1][0], y_new[k][0][1], y_new[k][1][1], size, img, base_colour)
    im = plt.imshow(img)
    frames.append([im])

print('Frames creation finished.')

# gif animation creation
ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=30)
ani.save("threeangle.gif", writer=writer)

plt.imshow(img)
plt.show()
