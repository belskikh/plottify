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


def draw_rect(ax, coords, c='white', lw=2):
    '''
    Draws a rectangle on ax and returns patch object
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param coords: tuple(list(int)) top-left and bottom-right rect corner coords in pixels
    :param c: (str) color of rectangle
    :param lw: (float) or (int) width of line
    :return: patch object
    '''
    patch = ax.add_patch(patches.Rectangle(coords[0], coords[1], fill=False, edgecolor=c, lw=lw))
    return patch


def draw_text(ax, coords, txt, sz=14, c='white', val='top', hal='left'):
    '''
    Draws a text on ax and returns text object
    :param ax: (matplotlib.axes._subplots.AxesSubplot) object
    :param coords: tuple(int) coordinates of top-left corner of text object
    :param txt: (str) text to draw
    :param sz: (int) font size
    :param c: (str) color
    :param val: (str) vertical alignment
    :param hal: (str) horizontal alignment
    :return: text object
    '''
    text = ax.text(*coords, txt,
                   fontsize=sz, color=c, verticalalignment=val, horizontalalignment=hal, weight='bold')
    return text

##############################
####### HELP FUNCTIONS #######
##############################

def geojson_to_plt(geojson, imsize):
    '''
    Transforms geojson polygon relative coordinates to matplotlib absolute coordinates (top-left, bot-right)

    :param geojson: (list(list(float))) coordinates of polygon in geojson format
    :param imsize: tuple(int) height and width of an image
    :return: tuple(tuple(int, int)) rectangle coordinates in pixels in (top-left, bot-right) format
    '''
    geojson = copy.deepcopy(geojson)
    # scale by image size
    for point in geojson:
        point[0] = int(point[0] * imsize[0])
        # flip y coord
        point[1] = int((1 - point[1]) * imsize[1])

    # sort to find top left and bottom right corners
    geojson = sorted(geojson, key=lambda point: (point[0] + point[1]))
    top_left = geojson[0]
    bottom_right = geojson[-1]
    return (top_left, bottom_right)

