import cv2

import itertools
import copy

from matplotlib import pyplot as plt
from matplotlib import (patheffects,
                        patches)

##############################
####### DRAW FUNCTIONS #######
##############################
def show_img(im, ax=None, figsize=None, show_axs=False):
    ''' Plots an image on ax object. If no ax object was passed, it creates an ax object
    :param im: (np.array) RGM image in np.array format
    :param ax: (matplotlib.axes._subplots.AxesSubplot)
    :param figsize: tuple(int) if ax is not None it will specify figure size
    :param show_axs: (bool) show or not pixel coordinates on plot
    :return: (matplotlib.axes._subplots.AxesSubplot) object
    '''
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)

    ax.get_xaxis().set_visible(show_axs)
    ax.get_yaxis().set_visible(show_axs)
    return ax


def draw_outline(o, lw, c='black'):
    '''
    Draws outline around object
    :param o: matplotlib object. ex: pathc, text etc.
    :param lw: (float) or (int) width of line
    :param c: (str) color of line
    :return: None
    '''
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground=c), patheffects.Normal()])


def draw_rect(ax, coords, c='white', lw=2, outline=True):
    '''
    Draws a rectangle on ax and returns patch object
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param coords: tuple(list(int)) top-left and bottom-right rect corner coords in pixels
    :param c: (str) color of rectangle
    :param lw: (float) or (int) width of line
    :param outline: (bool) draw or not black outline with lw=4
    :return: patch object
    '''
    patch = ax.add_patch(patches.Rectangle(coords[0], *coords[1], fill=False, edgecolor=c, lw=lw))
    if outline:
        draw_outline(patch, 4, c='black')
    return patch


def draw_text(ax, coords, txt, sz=14, c='white', val='top', hal='left', outline=True):
    '''
    Draws a text on ax and returns text object
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param coords: tuple(int) coordinates of top-left corner of text object
    :param txt: (str) text to draw
    :param sz: (int) font size
    :param c: (str) color
    :param val: (str) vertical alignment
    :param hal: (str) horizontal alignment
    :param outline: (bool) draw or not black outline with lw=4
    :return: text object
    '''
    text = ax.text(*coords, txt,
                   fontsize=sz, color=c, verticalalignment=val, horizontalalignment=hal, weight='bold')
    if outline:
        draw_outline(text, 4, c='black')
    return text


def plot_row(ax, objects, plot_func):
    '''
    Plots a row of objects with plot_func
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param objects: list of objects to plot
    :param plot_func: functions that will handle plotting of an object
    :return:
    '''
    for i, o in enumerate(objects):
        plot_func(o, ax[i % len(objects)])


def plot_grid(objects, plot_func, ncols=4, fig_width=18):
    '''
    Creates a grid of ncols width and len(objects)//ncols height and plots objects with plot_func
    :param objects (list): list of objects to plot
    :param plot_func: function that will handle plotting of an object
    :param ncols (int): number of columns in a grid
    :param fig_width: width of a figure
    :return:
    '''
    nrows = len(objects) // ncols + bool(len(objects) % ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(fig_width, nrows*5))
    # dummy workaround in the case of one-row grid
    if len(objects) <= ncols:
        ax = [ax]

    for i in range(nrows):
        slc = objects[i*ncols: i*ncols+ncols]
        plot_row(ax[i], slc, plot_func)
    plt.tight_layout()
    plt.show()

##############################
####### HELP FUNCTIONS #######
##############################

def geojson_to_plt(geojson, imsize):
    '''
    Transforms geojson polygon relative coordinates to matplotlib absolute coordinates ([x, y], [width, height])

    :param geojson: (list(list(float))) coordinates of polygon in geojson format
    :param imsize: tuple(int) height and width of an image
    :return: tuple(tuple(int, int)) rectangle coordinates in pixels in (top-left, bot-right) format
    '''
    geojson = copy.deepcopy(geojson)
    # scale by image size
    for point in geojson:
        point[0] = int(point[0] * imsize[1])  # horizontal coord
        point[1] = int((point[1]) * imsize[0])  # vertical coord

    # sort to find top left and bottom right corners
    geojson = sorted(geojson, key=lambda point: (point[0] + point[1]))
    top_left = geojson[0]
    bottom_right = geojson[-1]
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return (top_left, [width, height])


def open_img(impath):
    img = cv2.imread(impath, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:, :, ::-1]
    return img

