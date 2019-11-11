from typing import List, Tuple, Union

import cv2
import numpy as np


def open_img(impath: str,
             ignore_orientation=False) -> np.ndarray:
    if ignore_orientation:
        binary_flag =  cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
    else:
        binary_flag = cv2.IMREAD_COLOR
    img = cv2.imread(impath, binary_flag)
    if img is not None:
        img = img[:, :, ::-1]
    else:
        raise Exception(f"There is no such image '{impath}'")
    return img


def resize_keep_ratio(img: np.ndarray, 
                      max_size: int) -> np.ndarray:
    height, width = img.shape[:2]
    ratio = height / width
    size = None
    if height > width:
        size = (int(max_size / ratio), max_size)
    else:
        size = (max_size, int(max_size * ratio))

    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def crop_img(img: np.ndarray,
             imshape: Union[List, Tuple]) -> np.ndarray:
    top_pad = (img.shape[0] - imshape[0]) // 2
    left_pad = (img.shape[1] - imshape[1]) // 2

    return img[top_pad: imshape[0] + top_pad, left_pad: imshape[1] + left_pad]


def bbox_rel_to_abs(bbox: List[float],
                    imshape: Union[List, Tuple]) -> np.ndarray:
    '''transforms relative bbox coordinates to absolute'''
    bbox[0] = int(bbox[0] * imshape[1])  # resize x
    bbox[2] = int(bbox[2] * imshape[1])  # resize width

    bbox[1] = int(bbox[1] * imshape[0])  # resize y
    bbox[3] = int(bbox[3] * imshape[0])  # resize height

    return bbox
