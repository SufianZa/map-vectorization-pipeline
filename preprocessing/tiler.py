"""
This script generates training data for contour detection by extracting
corresponding tiles from the handdrawings and contour maps.
"""
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
source = "data"
map_source = join(source, "originale")
slice_source = join(source, "slices")
originalPath = os.path.abspath('data/originale')
tiles_dest = 'D:\Master\\dataset'

tile_size = 256  # size of tiles in pixels
tiles_per_map = 500  # how many tiles to extract from each map
chunk_size = 50  # maximum number of tiles per batch/hdf5 file

for map_num in tqdm(range(1,21)):
    filename = '{}.tif'.format(map_num)
    map_im = np.array(Image.open(os.path.join(originalPath, filename)))[:, :, :3]  # read image
    contour_image = np.array(Image.open(os.path.join(slice_source, filename + '-1.png')))[:, :, 0]
    sampling_weights = gaussian_filter(
        contour_image[tile_size//2:-tile_size//2,
                      tile_size//2:-tile_size//2,
                     ].astype(np.float),
        sigma=50., truncate=2.
    )
    inverse = 1 - contour_image/np.max(contour_image)
    dt = distance_transform_edt(inverse.astype(np.float))
    dt[dt > 50.] = 50.
    dt /= 50.
    linear = np.cumsum(sampling_weights)
    linear /= linear[-1]
    indices = np.searchsorted(linear,np.random.random_sample(tiles_per_map), side='right')
    train, val = train_test_split(indices, test_size=0.25, shuffle=True)
    for i, idx in enumerate(train):
        x = idx % sampling_weights.shape[1]
        y = idx // sampling_weights.shape[1]
        m_tile = map_im[y:y + tile_size, x:x + tile_size]
        w_tile = dt[y:y + tile_size, x:x + tile_size]
        Image.fromarray(m_tile).save(
            join(tiles_dest +'\\train\\inputs', filename+"-{}.png".format(i))
        )
        Image.fromarray((w_tile * 255).astype('uint8')).save(
            join(tiles_dest +'\\train\\labels', filename+"-{}.png".format(i))
        )
    for i, idx in enumerate(val):
        x = idx % sampling_weights.shape[1]
        y = idx // sampling_weights.shape[1]
        m_tile = map_im[y:y + tile_size, x:x + tile_size]
        w_tile = dt[y:y + tile_size, x:x + tile_size]
        Image.fromarray(m_tile).save(
            join(tiles_dest +'\\val\\inputs', "map_"+filename+"-{}.png".format(i))
        )
        Image.fromarray((w_tile * 255).astype('uint8')).save(
            join(tiles_dest +'\\val\\labels', "label_"+filename+"-{}.png".format(i))
        )



