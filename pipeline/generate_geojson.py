import geojson
import numpy as np
from shapely_geojson import dump, Feature, FeatureCollection
import cv2


def generate_GeoJSON(polygon_points, contours_feature, labels):
    """
    This method generates feature collection for geojson using polygons and points
    :param polygon_points: dictionary includes the keys:
        "points"                simple representation of sequence of points
        "shapely_obj"           polygons as shapely objects
        "neighboring_labels"    list of the labels of the neighboring polygons
        "properties"            the classification of Map Object Classifier
    :param contours_feature: shapely polygon of the contours
    :param labels: the labels image
    """
    mask = labels.copy()
    mask[mask > 0] = 255
    contours_mask, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    features = []
    f_pts = []
    for poly_p in polygon_points.values():
        properties = poly_p['properties']
        if poly_p['shapely_obj'].type == 'Polygon' or poly_p['shapely_obj'].type == 'MultiPolygon':
            feature = Feature(poly_p['shapely_obj'], properties=properties)
            features.append(feature)
            f_contours = [np.squeeze(con.astype(float)).tolist() for con in np.array([poly_p['points']])]
            c_obj = geojson.Polygon(f_contours)
            f_pts.append(geojson.Feature(geometry=c_obj, properties=properties))
    if contours_feature:
        features.append(Feature(contours_feature,
                                properties=dict(type='contours', fill='#FFFFFF')))  # add contours_feature to geo_json
    feature_gis = FeatureCollection(features)
    features_pts = geojson.FeatureCollection(f_pts)
    return feature_gis, features_pts
