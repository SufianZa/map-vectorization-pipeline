import threading
from pathlib import Path

from pipeline.deep_distance_transfrom import DeepDistanceTransform
import numpy as np
from PIL import Image
import geojson
import numpy as np
from shapely_geojson import dump
import global_variables
from pipeline.generate_geojson import generate_GeoJSON
from pipeline.map_object_classifier import MapObjectsClassifier
from pipeline.post_processing import post_processing
from pipeline.watershed_segmentation import watershed_transform

# initialize a Map Objects Classifier
map_obj_classifier = MapObjectsClassifier()


class PipelineThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.status = 'started'

    def run(self, path, id, para):
        self.run_pipeline(path, id, **para)

    def get_status(self):
        return self.status

    def run_pipeline(self, path, id, webapp=False, method=1, factor=2.85, extract_contour=False, sensitivity=0.15,
                     douglas=4, evaluator=None):
        global map_obj_classifier
        self.status = 'Approximate Euclidean Distance Transformation'
        # load image
        image = np.array(Image.open(path))[:, :, :3]
        edt_estimator = DeepDistanceTransform()

        # estimate Euclidean distance transformation
        edt_map = edt_estimator.estimate_full_map(image, trim=80)

        self.status = 'Watershed Segmentation'
        # watershed segmentation with estimated distance transformation

        labels = watershed_transform(image, edt_map, sensitivity)
        self.status = 'Post-processing and Object Classification'

        # post-processing module
        polygons, contours_feature = post_processing(labels, map_obj_classifier, image,
                                                     method=method, param=factor, douglas=douglas,
                                                     contours_extract=extract_contour)
        self.status = 'Generate GeoJSON Files'
        # extract GeoJSON file
        gqis_geojson, geojson_file = generate_GeoJSON(polygons, contours_feature, labels)

        if evaluator:
            mask = labels
            mask[mask > 0] = 255
            evaluator.evaluate(polygons, path=path, mask=mask)

        if webapp:
            with open(Path(global_variables.temp_path, id + '_gis.geojson'), 'w') as f:
                geojson.dump(gqis_geojson, f)
            with open(Path(global_variables.temp_path, id + '_simple.geojson'), 'w') as f:
                geojson.dump(geojson_file, f)
        else:
            with open(global_variables.geojson_path.format('QGIS_' + 'test.geojson'), 'w') as f:
                dump(gqis_geojson, f, indent=2)
            with open(global_variables.geojson_path.format('simple_' + 'test.geojson'), 'w') as f:
                geojson.dump(geojson_file, f)
        self.status = 'Done'


if __name__ == '__main__':
    # run_pipeline('X:\map-vectorization-pipeline\dataset\\test_maps\\5.tif')
    pass
