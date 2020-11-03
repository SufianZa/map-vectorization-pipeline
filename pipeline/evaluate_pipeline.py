from pipeline.deep_distance_transfrom import DeepDistanceTransform
import numpy as np
from PIL import Image

from pipeline.generate_geojson import generate_GeoJSON
from pipeline.map_object_classifier import MapObjectsClassifier
from pipeline.post_processing import post_processing
from pipeline.watershed_segmentation import watershed_transform

if __name__ == '__main__':
    # load image
    image = np.array(Image.open('X:\map-vectorization-pipeline\dataset\\test_maps\\5.tif'))[:, :, :3]
    edt_estimator = DeepDistanceTransform()

    # estimate Euclidean distance transformation
    edt_map = edt_estimator.estimate_full_map(image, trim=80)

    # watershed segmentation with estimated distance transformation
    labels = watershed_transform(image, edt_map)

    # post-processing module
    polygons, contours_feature = post_processing(labels, method=1, param=2.85, douglas=4, contours_extract=True)

    # initial Map Objects Classifier
    map_obj_classifier = MapObjectsClassifier(image)

    # extract GeoJSON file
    generate_GeoJSON(map_obj_classifier, polygons, contours_feature, labels)
