import numpy as np


def cropImage(label_segmented):
    true_points = np.argwhere(label_segmented)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    out = label_segmented[top_left[0]:bottom_right[0] + 1,
          top_left[1]:bottom_right[1] + 1]
    return out, top_left, bottom_right


def glueImage(inner_region_closed, label_segmented, top, bottom):
    label_segmented[top[0]:bottom[0] + 1, top[1]:bottom[1] + 1] = inner_region_closed
    return label_segmented


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def mad(arr, axis=0):
    med = np.median(arr, axis=axis)
    return np.median(np.abs(arr - med), axis=axis)
