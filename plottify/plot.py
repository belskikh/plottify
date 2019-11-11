from typing import Callable, Dict, List, Iterable, Optional, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib import patheffects, patches
import numpy as np

def show_img(im: np.ndarray,
             ax: Optional[plt.Figure] = None,
             figsize: Optional[Tuple[int, int]] = None,
             show_axs: bool = False) -> plt.Figure:
    ''' Plots an image on ax object. If no ax object was passed, 
    it creates an ax object
    :param im: (np.array) RGM image in np.array format
    :param ax: (matplotlib.axes._subplots.AxesSubplot)
    :param figsize: tuple(int) if ax is not None it will specify figure size
    :param show_axs: (bool) show or not pixel coordinates on plot
    :return: (matplotlib.axes._subplots.AxesSubplot) object
    '''
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)

    ax.get_xaxis().set_visible(show_axs)
    ax.get_yaxis().set_visible(show_axs)
    return ax


def draw_outline(o,
                 lw: Union[float, int], 
                 c: str = 'black'):
    '''
    Draws outline around object
    :param o: matplotlib object. ex: pathc, text etc.
    :param lw: (float) or (int) width of line
    :param c: (str) color of line
    :return: None
    '''
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground=c), patheffects.Normal()])


def draw_rect(ax: plt.Figure,
              coords: Tuple[List[int], List[int]], 
              c: str = 'white',
              lw: Union[float, int] = 2,
              outline: bool = True):
    '''
    Draws a rectangle on ax and returns patch object
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param coords: tuple(list(int)) top-left and bottom-right 
        rect corner coords in pixels
    :param c: (str) color of rectangle
    :param lw: (float) or (int) width of line
    :param outline: (bool) draw or not black outline with lw=4
    :return: patch object
    '''
    patch = ax.add_patch(patches.Rectangle(coords[0], *coords[1], 
                         fill=False, edgecolor=c, lw=lw))
    if outline:
        draw_outline(patch, 4, c='black')
    return patch


def draw_text(ax: plt.Figure,
              coords: Tuple[List[int], List[int]],
              txt: str,
              sz: int = 14, 
              c: str = 'white',
              val: str = 'top',
              hal: str = 'left', 
              outline: bool = True):
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
                   fontsize=sz, color=c, verticalalignment=val, 
                   horizontalalignment=hal, weight='bold')
    if outline:
        draw_outline(text, 4, c='black')
    return text


def plot_row(ax: plt.Figure, 
             objects: Iterable,
             plot_func: Callable):
    '''
    Plots a row of ofrom . import *bjects with plot_func
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param objects: list of objects to plot
    :param plot_func: functions that will handle plotting of an object
    :return:
    '''
    for i, o in enumerate(objects):
        plot_func(o, ax[i % len(objects)])


def plot_grid(objects: Iterable,
              plot_func: Callable,
              ncols: int = 4,
              fig_width: int = 18):
    '''
    Creates a grid of ncols width and len(objects)//ncols height and plots 
    objects with plot_func
    :param objects (list): list of objects to plot
    :param plot_func: function that will handle plotting of an object
    :param ncols (int): number of columns in a grid
    :param fig_width: width of a figure
    :return:
    '''
    nrows = len(objects) // ncols + bool(len(objects) % ncols)
    _, ax = plt.subplots(nrows, ncols, figsize=(fig_width, nrows*5))
    # dummy workaround in the case of one-row grid
    if len(objects) <= ncols:
        ax = [ax]

    for i in range(nrows):
        slc = objects[i*ncols: i*ncols+ncols]
        plot_row(ax[i], slc, plot_func)
    plt.tight_layout()
    plt.show()
