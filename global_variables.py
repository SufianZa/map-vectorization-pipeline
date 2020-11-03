"""
This code outline global parameters such as paths for the pipeline modules, which also contains classes
and their corresponding colors given in the dataset
"""
import functools
import operator
from pathlib import Path, PurePath
import numpy as np
from pipeline.utils import RGB2HEX

# project path
abs_path = Path(__file__).parent

# full input raster maps
train_full_maps = Path(abs_path, 'dataset', 'train_maps')

# full input vectorized maps
vector_full_images = Path(abs_path, 'dataset', 'corrected')

# maps excluded from the training to use for testing
test_full_maps = Path(abs_path, 'dataset', 'test_maps')

# the path of the extracted classes as binary masks check preprocessing\pre_processing.py
classes_path = Path(abs_path, 'dataset', 'classes')
classes_path.mkdir(parents=True, exist_ok=True)

# the dictionary paths for train, val and test of the pairs of image with the fixed size
train_test_val_path = dict(
    train=dict(x=Path(abs_path, 'train_test_val', 'train', 'input', 'inputs'),
               y=Path(abs_path, 'train_test_val', 'train', 'label', 'labels')),
    val=dict(x=Path(abs_path, 'train_test_val', 'val', 'input', 'inputs'),
             y=Path(abs_path, 'train_test_val', 'val', 'label', 'labels')),
    test=dict(x=Path(abs_path, 'train_test_val', 'test', 'input', 'inputs'),
              y=Path(abs_path, 'train_test_val', 'test', 'label', 'labels')))

# the colors as they are defined in the raw vectorized maps
class_colors = dict(contours=[[127, 127, 127]],
                    background=[[255, 255, 255], [153, 255, 178], [191, 254, 178]],
                    building=[[255, 191, 217], [255, 127, 204]],
                    water=[[204, 255, 255]], )

# the colors as a list
colors = functools.reduce(operator.iconcat, class_colors.values(), [])

# the list of target classes
target_classes = list(class_colors.keys())
# the colors used for pca visualization of each class
visualize_colors = dict(background=[127, 127, 127], building=[255, 127, 204], water=[74, 210, 245])

# the path of weights
weights_path = Path(abs_path, 'weights')
weights_path.mkdir(parents=True, exist_ok=True)

# the path of the geojson
geojson_path = Path(abs_path, 'geoJSON')
geojson_path.mkdir(parents=True, exist_ok=True)
geojson_path = str(PurePath(geojson_path, '{}.geojson'))

# the path of the temporary directory for the web application
temp_path = Path(abs_path, 'templates', 'static', 'temp')
temp_path.mkdir(parents=True, exist_ok=True)


# the colors used for visualize each class after classification
visualization_colors = {'background': RGB2HEX([127, 127, 127]),
                        'water': RGB2HEX([74, 210, 245]),
                        'unknown': RGB2HEX([44, 194, 67]),
                        'building': RGB2HEX([255, 127, 204])}

# the name of each used feature with the respect of the order defined in
# pipeline\feature_extraction.py "get_spectral_features" and "get_shape_features"
spectral_shape_features_name = np.array(['$Median_{h}$', '$Median_{s}$', '$Median_{v}$',
                                         '$MAD_{h}$', '$MAD_{s}$', '$MAD_{v}$',
                                         '$Mean_{h}$', '$Mean_{s}$', '$Mean_{v}$',
                                         '$Skew_{h}$', '$Skew_{s}$', '$Skew_{v}$',
                                         '$Std_{h}$', '$Std_{s}$', '$Std_{v}$',
                                         'rectangularity', 'solidity', 'aspect_ratio', 'elongation', 'circularity',
                                         'convexity',
                                         'relative_area'])

# the array that includes and/concludes features
cherry_pick_features = np.arange(len(spectral_shape_features_name))

# the indices of the spectral and shape features
spectral_index = np.arange(0, 15)
shape_index = np.arange(15, len(spectral_shape_features_name))

# the indices of SELECTED shape and spectral features
selected_spectral_features = [0, 1, 2, 6, 7, 8, 12, 14]
selected_shape_features = [15, 19, 20, 21]

