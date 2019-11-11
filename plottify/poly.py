import copy
from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from PIL import Image, ImageDraw

from .plot import show_img


def geojson_to_plt(geojson: List[List[int]], 
                   imsize: Union[List, Tuple]
    ) -> Tuple[List[int], List[int]]:
    '''
    Transforms geojson polygon relative coordinates to matplotlib absolute 
    coordinates ([x, y], [width, height])

    :param geojson: (list(list(float))) coordinates of polygon in geojson format
    :param imsize: tuple(int) height and width of an image
    :return: tuple(tuple(int, int)) rectangle coordinates in pixels in 
        (top-left, bot-right) format
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


def plot_polygons(img: np.ndarray,
                  poly_dict: Dict,
                  alpha: float = 0.5,
                  ax: Optional[plt.Figure] = None, 
                  figsize: Optional[Tuple[int, int]] = None,
                  show_axs: bool = False,
                  convert: bool = False):
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    show_img(img, ax, figsize, show_axs)
    patches = []
    colors = []

    labels = list(poly_dict.keys())
    for l in labels:
        color = 100*np.random.rand()
        for polygons in poly_dict[l]:
            for poly in polygons:
                if poly:
                    if convert:
                        poly = np.array(convert_poly(poly, img.shape))
                    patches.append(Polygon(poly, True))
            colors.append(color)
    patch_collection = PatchCollection(patches, alpha=alpha)
    patch_collection.set_array(np.array(colors))
    ax.add_collection(patch_collection)


def plot_bboxes(img: np.ndarray,
                bbox_dict: Dict, 
                colormap: Dict, 
                alpha: float = 0.5,
                ax: Optional[plt.Figure] = None, 
                figsize: Optional[Tuple[int, int]] = None,
                show_axs: bool = False):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    show_img(img, ax, figsize, show_axs)
    patches = []
    colors = []

    labels = list(bbox_dict.keys())
    for l in labels:
        color = colormap[l]
        for poly in bbox_dict[l]:
            if poly:
                p = np.array(convert_poly(poly, img.shape))
                patches.append(Polygon(p, True))
            colors.append(color)
    patch_collection = PatchCollection(patches, alpha=alpha)
    patch_collection.set_array(np.array(colors))
    ax.add_collection(patch_collection)


def convert_poly(poly: List[float],
                 imshape: Union[List[int], Tuple[int, int]]) -> List[int]:
    return list(map(lambda p: (int(p[0]*imshape[1]), int(p[1]*imshape[0])),
                    poly))


def mask_from_polygon(polygon: List[int],
                      imshape: Union[List[int], Tuple[int, int]]
    ) -> np.ndarray:
    img = Image.new('L', (imshape[1], imshape[0]), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img)
