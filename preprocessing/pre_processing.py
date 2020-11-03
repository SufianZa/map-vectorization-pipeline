"""

This script performs pre-processing including maps registration, using k-means clustering of pre-defined colors to
extract classes of vectorized maps and generates training tiles pairs comprised of original map tiles data as input
data and Euclidean distance transformation as labels.

"""

import global_variables
from pathlib import Path, PurePath
import numpy as np
from scipy.cluster.vq import kmeans2 as KMeans
from skimage.morphology import label, remove_small_objects, skeletonize
from skimage.transform import estimate_transform, warp
import cv2
import os
from os.path import join
from math import ceil
from os import listdir

from tqdm import tqdm as tqdm
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing.image_registration import register_images


def extract_class_from_colors(map_name: str, vector_image=None):
    """
     Extracts classes using Kmeans clustering depending on pre-defined colors in global_variables
     then saves each class in global_variables.classes_path

    :param map_name: str
            the name to save each class as <map_name>-<class>.png
    :param vector_image: ndarray
            the registered vectroized ground truth image as np.array
    :returns ndarray
            the contour class image
    """
    vector_image = np.array(Image.open(
        os.path.join(global_variables.vector_full_images, '{}.tif'.format(map_name))))[:, :, :3]
    img_reshape = vector_image.reshape(-1, 3)
    _, parts = KMeans(data=img_reshape.astype(np.float), k=global_variables.colors)

    # merge classes check in global_variables.class_colors to see the color dictionary
    # assign background  1, 2, 3 into 1
    parts[parts == 1] = 1
    parts[parts == 2] = 1
    parts[parts == 3] = 1
    # assign building  4,5 into 2
    parts[parts == 4] = 2
    parts[parts == 5] = 2
    # assign water 6 into 3
    parts[parts == 6] = 3
    layers = np.moveaxis(
        np.eye(len(global_variables.target_classes))[parts].reshape(vector_image.shape[0], vector_image.shape[1],
                                                                    len(global_variables.target_classes)), -1, 0
    )

    # collect the rests of the contours from other classes
    for i in range(1, 4):
        removed_small = (remove_small_objects(label(layers[i], connectivity=1)) > 0).astype(bool)
        rest_contours = layers[i] - removed_small
        layers[i] = removed_small.astype(np.float)
        layers[0] += rest_contours

    # generate thinner binary contours
    layers[0] = skeletonize(layers[0])

    # save each class as <map_name>-<class>.png in global_variables.target_classes
    for i, target_class in enumerate(global_variables.target_classes):
        Image.fromarray(
            layers[i] * 255
        ).convert("RGB").save(join(global_variables.classes_path, "{}-{}.png".format(map_name, target_class)))

    return layers[0]


def generate_train_edt_data(input_img, map_name, contour_image=None, patch_size=256, patches_per_map=15):
    """
    Generates n pairs of tiles using the input image as training data and
     its registered contour image as labels after applying Euclidean distance transformation

    :param input_img: ndarray
        registered input_map image
    :param contour_img: ndarray
        registered contour image
    :param map_name: str
        the name of the map
    :param patches_per_map: int
        number of patches pairs to be extracted of the map
    :param patch_size: int
        the side length of the squared window of extracted patches pairs
    """
    if not contour_image:
        contour_image = np.array(
            Image.open(os.path.join(global_variables.classes_path, '{}-contours.png'.format(map_name))),
            dtype=np.float)[:, :, 0]

    # calculate normalized Euclidean distance transform matrix of the contour image
    inverse = 1 - contour_image / np.max(contour_image)
    dt = distance_transform_edt(inverse.astype(np.float))
    dt[dt > 50.] = 50.
    dt /= 50.

    # calculate most important indices of contours
    sampling_weights = gaussian_filter(
        contour_image[patch_size // 2:-patch_size // 2,
        patch_size // 2:-patch_size // 2,
        ].astype(np.float),
        sigma=50., truncate=2.
    )
    linear = np.cumsum(sampling_weights)
    linear /= linear[-1]
    indices = np.searchsorted(linear, np.random.random_sample(patches_per_map), side='right')

    # splitting the image into patches 80% train, 15% validation and 5% test
    train, val_test = train_test_split(indices, test_size=0.20, shuffle=True)
    val, test = train_test_split(val_test, test_size=0.25, shuffle=True)

    # saving image patches for each split in train, val and test directories
    for subset_name, subset in dict(train=train, val=val, test=test).items():
        for i, idx in enumerate(subset):
            x = idx % sampling_weights.shape[1]
            y = idx // sampling_weights.shape[1]
            input_patch = input_img[y:y + patch_size, x:x + patch_size]
            edt_patch = dt[y:y + patch_size, x:x + patch_size]
            assert input_patch.shape[0] == input_patch.shape[1] == edt_patch.shape[0] == edt_patch.shape[
                1] == patch_size
            global_variables.train_test_val_path[subset_name]['x'].mkdir(parents=True, exist_ok=True)
            global_variables.train_test_val_path[subset_name]['y'].mkdir(parents=True, exist_ok=True)
            Image.fromarray(input_patch).save(
                os.path.join(global_variables.train_test_val_path[subset_name]['x'],
                             map_name + "-{}.png".format(i)))
            Image.fromarray((edt_patch * 255).astype('uint8')).save(
                os.path.join(global_variables.train_test_val_path[subset_name]['y'], map_name + "-{}.png".format(i)))


def process_image(file_path, register=False, extract_class=False, train_data=False):
    """
    Automate the pre-processing operation in order to register, extract classes or/and generate train data.

    :param file_path: str
        the path of the input image in train/test folders
    :param register: bool
        flag to activate image registration
    :param extract_class: bool
        flag to activate image class extraction
    :param train_data: bool
        flag to activate image train data generation
    """
    filename = os.path.basename(file_path)
    map_name = filename.split('.')[0]
    input_image = np.array(Image.open(file_path))[:, :, :3]
    vector_path = PurePath(global_variables.vector_full_images, filename)
    vectorized_image = np.array(Image.open(str(vector_path)))[:, :, :3]
    contours = None

    # register images
    if register:
        input_image, vectorized_image = register_images(moving_image=input_image,
                                                        fixed_image=vectorized_image)
        Image.fromarray(input_image* 255).convert("RGB").save(str(file_path))

    # extract class images
    if extract_class:
        contours = extract_class_from_colors(map_name, vectorized_image)

    # generate patches of train and label images
    if train_data:
        generate_train_edt_data(input_image, map_name, contours)


if __name__ == '__main__':
    register = False
    extract_classes = True

    # print('preparing train images...')
    # for file_path in global_variables.train_full_maps.glob('*.*'):
    #     process_image(file_path, register=register, extract_class=extract_classes, train_data=True)

    print('preparing test images...')
    for file_path in global_variables.test_full_maps.glob('*.*'):
        process_image(file_path, register=register, extract_class=extract_classes,
                      train_data=False)
    print('pre-processing is done')
