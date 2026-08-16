import os

import cv2
import numpy as np


def colorize_mask(mask, palette):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, color in enumerate(palette):
        color_mask[mask == index] = color
    return color_mask


def save_colorized_mask(output_dir, filename, mask, palette):
    output_path = os.path.join(output_dir, f"pred_{filename}")
    if not cv2.imwrite(output_path, colorize_mask(mask, palette)):
        raise OSError(f"Unable to save visualization: {output_path}")
