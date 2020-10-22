"""
This script reads the rasterized digital maps, aligns them to the handdrawings and
stores every face type as a separate binary map.

The points used for alignment must be provided in the 'points' dictionary, same goes
for the sizes of the handdrawings.
"""
import os
import matplotlib.pyplot as plt
from os.path import exists, join
from PIL import Image
import numpy as np
from scipy.cluster.vq import kmeans2 as KMeans
from skimage.morphology import label, remove_small_objects, skeletonize
from skimage.transform import estimate_transform, warp
import cv2
source = "../data"
map_source = os.path.join(source, 'originale')
bnk_source = os.path.join(source, 'corrected')
slice_dest = os.path.join(source, 'slices')

colors = np.array(
    [
        [255, 255, 255],  # 0: background
        [127, 127, 127],  # 1: contours
        [191, 254, 178],  # 2: light green
        [153, 255, 178],  # 3: dark green
        [204, 255, 255],  # 4: water
        [255, 191, 217],  # 5: lighter buildings
        [255, 127, 204],  # 6: darker buildings
    ],
    dtype=np.float,
)

def read_bnk_from_rgba(filename: str):
    if not exists(filename):
        raise ValueError("File '{}' does not exist.".format(filename))
    im = np.array(Image.open(filename))[:, :, :3]
    buff = im.reshape(-1, 3)
    parts = KMeans(data=buff.astype(np.float), k=colors)[1].astype(np.int)
    num_labels = np.max(parts) + 1
    layers = np.moveaxis(
        np.eye(num_labels)[parts].reshape(im.shape[0], im.shape[1], num_labels), -1, 0
    )


    for i in range(2, 7):
        buff = remove_small_objects(label(layers[i], neighbors=4), min_size=30) > 0
        diff = layers[i] - buff  # diff contains the small objects
        layers[i] = buff.astype(np.float)
        layers[1] += diff  # add small objects to contours
    return layers.astype(np.float)

vectorPath = os.path.abspath('../data/corrected')
if __name__ == "__main__":
    for root, subdirs, files in os.walk(vectorPath):
        for filename in files:
            if filename == '24_big.tif':
                p = os.path.join(vectorPath, filename)
                print(p)
                vectorized_image = cv2.imread(p, -1)
                image_array = read_bnk_from_rgba(p)
                # tform = estimate_transform(
                #     "affine", src=reg_points[map_no]["src"], dst=reg_points[36]["dst"]
                # )
                for i in range(1,7):
                    canvas = np.zeros((vectorized_image.shape[0], vectorized_image.shape[1]))
                    if i == 1:
                        canvas[: image_array.shape[1], : image_array.shape[2]] = skeletonize(
                            image_array[i]
                        )
                    else:
                        canvas[: image_array.shape[1], : image_array.shape[2]] = image_array[i]
                    Image.fromarray(
                                  canvas * 255
                                ).convert("RGB").save(os.path.join(slice_dest, filename+"-{}.png".format(i)))

