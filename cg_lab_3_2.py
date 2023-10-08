import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'



def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr

def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr

def to_proj_coords(x):
    r,c = x.shape
    x = np.concatenate([x, np.ones((1,c))], axis = 0)
    return x

def to_cart_coords(x):
    x = x[:-1]/x[-1]
    return x

pt0 = [50, 50]
pt1 = [50, 150]

x = np.array([pt0, pt1], dtype = np.float32).T
center = np.sum(x, axis=1)/2

x_proj = to_proj_coords(x)

N = 100 # frames count
size = 256
color = np.array([0, 0, 0], dtype=np.uint8)

frames = []
fig = plt.figure()
count = 0

for i in range(N):
    # get coords of transformed line
    ang = -i*2*np.pi/N
    count += 1
    T = shiftMatr(-center)
    M = shiftMatr([count, count])
    R = rotMatr(ang)
    x_proj_new = np.linalg.inv(T) @ M @ T @ np.linalg.inv(T) @ R @ T @ x_proj
    x_new = to_cart_coords(x_proj_new)

    # draw line
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):  # Изменим цвет фона
        for j in range(size):
            for k in range(3):
                img[i][j][k] = 255
    line_points_count = np.int32(np.max(np.abs(x_new[:,0] - x[:,1])) + 1)
    t = np.linspace(0,1,line_points_count)
    a = x_new[:, 0].reshape(-1, 1)
    b = x_new[:, 1].reshape(-1, 1)
    t = t.reshape(1, -1)
    line_points = (1 - t) * a + t * b  # not clean lines, use Bresehnam instead
    line_points = np.int32(np.round(line_points))
    img[line_points[0], line_points[1]] = color
    im = plt.imshow(img)
    frames.append([im])


print('Frames creation finshed.')


#mp4 animation creation
#ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('line.mp4', writer)
#ani.save('simple_animation.mp4')


# gif animation creation
ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=30)
ani.save("line.gif", writer=writer)


plt.show()