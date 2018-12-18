from pdb import set_trace as st

import cv2

import copy
import itertools

from collections import Iterable

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import (patheffects,
                        patches)
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from PIL import Image, ImageDraw


__all__ = ["show_img", "draw_outline", "draw_rect", "draw_text", "plot_grid",
           "rle_from_masked", "mask_from_rle", "geojson_to_plt",
           "open_img", "resize_keep_ratio", "crop_img", "bbox_rel_to_abs",
           "plot_polygons", "plot_bboxes", "mask_from_polygon"]

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
####### RLE FUNCTIONS ########
##############################

def rle_from_masked(img):
    shape = img.shape
    RED = 0
    mask = (img[:,:,RED] >= 254).astype(np.int)
    is_line = False
    idx = None
    pixels = None
    rle = []
    for i, elem in enumerate(mask.reshape(-1)):
        if elem == 1:
            if not is_line:
                is_line = True
                idx = i
                pixels = 0
            pixels += 1
        else:
            if is_line:
                is_line = False
                rle.append((idx, pixels))
    return rle


def mask_from_rle(rle, imshape):
    mask = np.zeros(imshape[0] * imshape[1], dtype=np.uint8)
    for start, length in rle:
        mask[start: start+length] = 1
    return mask.reshape(imshape[:2])


##############################
####### HELP FUNCTIONS #######
##############################

def open_img(impath, ignore_orientation=False):
    if ignore_orientation:
        binary_flag =  cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
    else:
        binary_flag = cv2.IMREAD_COLOR
    img = cv2.imread(impath, binary_flag)
    if img is not None:
        img = img[:,:,::-1]
    else:
        raise Exception(f"There is no such image '{impath}'")
    return img


def resize_keep_ratio(img, max_size):
    height, width = img.shape[:2]
    ratio = height / width
    size = None
    if height > width:
        size = (int(max_size / ratio), max_size)
    else:
        size = (max_size, int(max_size * ratio))

    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def crop_img(img, imshape):
    top_pad = (img.shape[0] - imshape[0]) // 2
    left_pad = (img.shape[1] - imshape[1]) // 2

    return img[top_pad: imshape[0] + top_pad, left_pad: imshape[1] + left_pad]


def bbox_rel_to_abs(bbox, imshape):
    '''transforms relative bbox coordinates to absolute'''
    bbox[0] = int(bbox[0] * imshape[1])  # resize x
    bbox[2] = int(bbox[2] * imshape[1])  # resize width

    bbox[1] = int(bbox[1] * imshape[0])  # resize y
    bbox[3] = int(bbox[3] * imshape[0])  # resize height

    return bbox


##############################
########## POLYGONS ##########
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


def plot_polygons(img, poly_dict, alpha=0.5, ax=None, figsize=None,
                  show_axs=False):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    show_img(img, ax, figsize, show_axs)
    patches = []
    colors = []

    labels = list(poly_dict.keys())
    st()
    for l in labels:
        color = 100*np.random.rand()
        for polygons in poly_dict[l]:
            for poly in polygons:
                if poly:
                    # st()
                    p = np.array(convert_poly(poly, img.shape))
                    patches.append(Polygon(p, True))
            colors.append(color)
    patch_collection = PatchCollection(patches, alpha=alpha)
    patch_collection.set_array(np.array(colors))
    ax.add_collection(patch_collection)


def plot_bboxes(img, bbox_dict, colormap, alpha=0.5, ax=None, figsize=None,
                show_axs=False):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    show_img(img, ax, figsize, show_axs)
    patches = []
    colors = []

    labels = list(bbox_dict.keys())
    # st()
    for l in labels:
        color = colormap[l]
        for poly in bbox_dict[l]:
            if poly:
                # st()
                p = np.array(convert_poly(poly, img.shape))
                patches.append(Polygon(p, True))
            colors.append(color)
    patch_collection = PatchCollection(patches, alpha=alpha)
    patch_collection.set_array(np.array(colors))
    ax.add_collection(patch_collection)


def convert_poly(poly, imshape):
    # if not isinstance(poly[0], Iterable):
    #     poly = [poly]
    return list(map(lambda p: (int(p[0]*imshape[1]), int(p[1]*imshape[0])),
                    poly))


def mask_from_polygon(polygon, imshape):
    img = Image.new('L', (imshape[1], imshape[0]), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img)