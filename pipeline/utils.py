import numpy as np


def crop_image(full_image):
    true_points = np.argwhere(full_image)
    top_pad = true_points.min(axis=0)
    bottom_pad = true_points.max(axis=0)
    cropped = full_image[top_pad[0]:bottom_pad[0] + 1, top_pad[1]:bottom_pad[1] + 1]
    return cropped, top_pad, bottom_pad


def glue_image(small_image, full_image, top_pad, bottom_pad):
    full_image[top_pad[0]:bottom_pad[0] + 1, top_pad[1]:bottom_pad[1] + 1] = small_image
    return full_image


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def mad(arr, axis=0):
    med = np.median(arr, axis=axis)
    return np.median(np.abs(arr - med), axis=axis)
