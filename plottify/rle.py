from typing import List, Tuple, Union

import numpy as np


def rle_from_masked(img: np.ndarray) -> List[int]:
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


def mask_from_rle(rle: List[int],
                  imshape: Union[List, Tuple]) -> np.ndarray:
    mask = np.zeros(imshape[0] * imshape[1], dtype=np.uint8)
    for start, length in rle:
        mask[start: start+length] = 1
    return mask.reshape(imshape[:2])
