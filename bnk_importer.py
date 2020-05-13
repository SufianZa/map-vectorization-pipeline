"""
This script reads the rasterized digital maps, aligns them to the handdrawings and
stores every face type as a separate binary map.

The points used for alignment must be provided in the 'points' dictionary, same goes
for the sizes of the handdrawings.
"""
from os.path import exists, join
from PIL import Image
import numpy as np
from scipy.cluster.vq import kmeans2 as KMeans
from skimage.morphology import label, remove_small_objects, skeletonize
from skimage.transform import estimate_transform, warp

source = "data/Raster"
map_source = join(source, "Urkarten")
bnk_source = join(source, "corrected")
slice_dest = join(source, "slices")

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

sizes = {36: [7302, 7842]}
maps = list(sizes.keys())

# these are the points needed to register both inputs 
points = {
    36: {
        "src": np.array(
            [
                [966, 291],
                [792, 399],
                [560, 418],
                [334, 876],
                [217, 1229],
                [930, 1146],
                [859, 1718],
                [1001, 1691],
                [1402, 1653],
                [905, 2097],
                [1371, 1992],
                [1778, 1968],
                [2052, 1758],
            ]
        ),
        "dst": np.array(
            [
                [2858, 530],
                [2266, 892],
                [1485, 953],
                [722, 2498],
                [316, 3692],
                [2730, 3408],
                [2488, 5343],
                [2958, 5250],
                [4315, 5127],
                [2640, 6621],
                [4208, 6259],
                [5576, 6192],
                [6504, 5494],
            ]
        ),
    }
}


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


if __name__ == "__main__":
    for map_no in maps:
        image_array = read_bnk_from_rgba(
            join(bnk_source, "Flur{}.tif".format(map_no))
        )
        tform = estimate_transform(
            "affine", src=points[map_no]["src"], dst=points[36]["dst"]
        )
        for i in range(7):
            canvas = np.zeros(sizes[map_no])
            if i == 1:
                canvas[: image_array.shape[1], : image_array.shape[2]] = skeletonize(
                    image_array[i]
                )
            else:
                canvas[: image_array.shape[1], : image_array.shape[2]] = image_array[i]
            Image.fromarray(
                warp(canvas, tform.inverse, order=0, mode="constant", cval=0) * 255
            ).convert("RGB").save(join(slice_dest, "Flur{}-{}.png".format(map_no, i)))

