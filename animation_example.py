import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import time
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

def create_image(heigh, widht, background_color):
    img = np.zeros((heigh, widht, 4), np.uint8)
    img[:, :, :3] = background_color
    img[:, :, 3] = 255

    return img

def set_color(img, x, y, color):

    img[x, y, :3] = color

    return img

def create_simple_animation():
    h = 1024
    w = 1024
    black = np.array([0, 0, 0], np.uint8)

    img = create_image(h, w, black)

    frames_count = 100
    points_num = 100

    fig = plt.figure(figsize=(15, 15))

    ims = []
    for i in range(frames_count):
        color = np.random.randint(0, 255, 3)
        x_inds = np.random.randint(0, h-1, points_num)
        y_inds = np.random.randint(0, w-1, points_num)
        img = set_color(img, x_inds, y_inds, color)
        im = plt.imshow(img, animated=True)
        ims.append([im])
    print('Frames creation finshed.')
    ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True, repeat_delay=5000)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('dynamic_images.mp4', writer)
    ani.save('simple_animation.mp4')

    plt.show()



if __name__ == '__main__':
    start = time.time()
    create_simple_animation()
    finish = time.time()
    print('Execution time: ', finish - start)
