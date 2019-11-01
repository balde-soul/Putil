# coding=utf-8

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import numpy as np


def plot_sequence_images(image_array):
    ''' Display images sequence as an animation in jupyter notebook
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    dpi = 72
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=33, repeat_delay=10, repeat=True)
    animation.FuncAnimation
    display(HTML(anim.to_html5_video()))
    pass


def plot_animation_function(xy_generator, frames=100, figsize=(6, 4), xlim=(0, 2), ylim=(0, 2), interval=20, blit=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(*args, **kwargs):
        x, y = next(xy_generator)
        line.set_data(x, y)
        return (line,)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=blit)
    display(HTML(anim.to_html5_video()))
    fig.delaxes(ax)
    pass
