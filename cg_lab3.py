import numpy as np
import matplotlib.pyplot as plt


def bVec():  # Вектор сдвига
    b = np.array([0, 0])
    return b

def rotMatr(ang):  # Поворот
    mtr = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return mtr

def diagMatr():  # Масштабирование
    mtr = np.array([[2, 0], [0, 2]])
    return mtr

def detPos():  # Пример матрицы
    mtr = np.array([[1.5, 1], [0.5, 1.2]])
    return mtr

def detNeg():  # Пример матрицы
    mtr = np.array([[0.5, 1], [1.5, -1]])
    return mtr

def to_proj_coords(x):  # Переход к проективным из афинных
    r,c = x.shape
    x = np.concatenate([x, np.ones((1,c))], axis = 0)
    return x

def to_cart_coords(x):  # В афинные координаты из проективных
    x = x[:-1]/x[-1]
    return x

pt0 = [1, 1]
pt1 = [3, 3]
pt2 = [5, 2]

x = np.array([pt0, pt1, pt2], dtype = np.float32).T  # T - транспонирование

x_proj = to_proj_coords(x)

# linear transform matrix
#a = rotMatr(np.pi/4) # change rotation angle
a = diagMatr()
#a = detPos()
#a = detNeg()

# shift vector
b = bVec() # change to non-zero

m = np.zeros((3,3))
m[:2,:2] = a
m[:2,-1] = b
m[-1,-1] = 1

x_new_proj = m @ x_proj  # Умножение двух матриц

x_new = to_cart_coords(x_new_proj)

# drawing
plt.figure()
# plt axes
plt.plot([-10, 10],[0,0], 'k')
plt.plot([0,0],[-10, 10], 'k')
# plot initial figure
plt.plot(x[0, [0,1,2,0]], x[1, [0,1,2,0]], 'r')
plt.plot(x[0,0], x[1,0], 'or')
plt.plot(x[0,1], x[1,1], 'og')
plt.plot(x[0,2], x[1,2], 'ob')
# plot transformed figure
plt.plot(x_new[0, [0,1,2,0]], x_new[1, [0,1,2,0]], 'g')
plt.plot(x_new[0,0], x_new[1,0], 'or')
plt.plot(x_new[0,1], x_new[1,1], 'og')
plt.plot(x_new[0,2], x_new[1,2], 'ob')

plt.show()


