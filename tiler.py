"""
This script generates training data for contour detection by extracting
corresponding tiles from the handdrawings and contour maps.
"""
from os.path import join
from math import ceil
from os import listdir
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import h5py

source = "data/Raster"
map_source = join(source, "Urkarten")
slice_source = join(source, "slices")
tiles_dest = join(source, "tiles")
hdf5_dest = tiles_dest
max_dist = 50.

maps = [36]
tile_size = 256  # size of tiles in pixels
tiles_per_map = 16384  # how many tiles to extract from each map
chunk_size = 1024  # maximum number of tiles per batch/hdf5 file

save_tiles_as_image = False
save_tiles_as_hdf5 = True

available_maps = [n for n in listdir(map_source)]


for map_no in maps:
    map_file = [n for n in available_maps if "Flur{}".format(map_no) in n][0]
    map_im = np.array(Image.open(join(map_source, map_file)))[:, :, :3]
    contour_image = np.array(Image.open(join(slice_source, "Flur{}-1.png".format(
        map_no))))[:, :, 0]
    sampling_weights = gaussian_filter(
        contour_image[tile_size//2:-tile_size//2,
                      tile_size//2:-tile_size//2,
                     ].astype(np.float),
        sigma=50., truncate=2.
    )
    inverse = 1 - contour_image/np.max(contour_image)
    dt = distance_transform_edt(inverse.astype(np.float))
    dt[dt > max_dist] = max_dist
    dt /= max_dist
    linear = np.cumsum(sampling_weights)
    linear /= linear[-1]
    for batch in range(int(ceil(tiles_per_map / chunk_size))):
        indices = np.searchsorted(linear,
                                  np.random.random_sample(
                                      (min(chunk_size,
                                           tiles_per_map - batch*chunk_size), )
                                  ),
                                  side='right')
        maps = []
        labels = []
        weights = []
        for i, idx in enumerate(indices):
            x = idx % sampling_weights.shape[1]
            y = idx // sampling_weights.shape[1]
            m_tile = map_im[y:y + tile_size, x:x + tile_size]
            c_tile = contour_image[y:y + tile_size, x:x + tile_size]
            w_tile = dt[y:y + tile_size, x:x + tile_size]
            if save_tiles_as_image:
                Image.fromarray(m_tile).save(
                    join(tiles_dest, "map_{}-{}.png".format(map_no,
                                                            batch*chunk_size + i))
                )
                Image.fromarray(c_tile).save(
                    join(tiles_dest, "cont_{}-{}.png".format(map_no,
                                                             batch*chunk_size + i))
                )
                Image.fromarray((w_tile * 255).astype('uint8')).save(
                    join(tiles_dest, "weight_{}-{}.png".format(map_no,
                                                               batch * chunk_size + i))
                )
            if save_tiles_as_hdf5:
                maps.append(m_tile.astype('float32')/np.max(m_tile))
                labels.append(c_tile.astype('float32')/np.max(c_tile))
                weights.append(w_tile.astype('float32'))

        if len(maps) > 0 & save_tiles_as_hdf5:
            if tiles_per_map > chunk_size:
                filename = "Flur{}-{}.hdf5".format(map_no, batch)
            else:
                filename = "Flur{}.hdf5".format(map_no)
            with h5py.File(join(tiles_dest, filename), "w") as f:
                f.create_dataset('data', data=np.moveaxis(
                    np.array(maps, dtype='float32'),
                    -1, 1),
                                 compression='gzip')
                f.create_dataset('weights', data=np.array(weights, dtype='float32')[
                                                 :, np.newaxis, :, :],
                                 compression='gzip')
                f.create_dataset('labels', data=np.array(labels, dtype='float32')[
                                                 :, np.newaxis, :, :],
                                 compression='gzip')
