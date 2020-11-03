import geojson
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image
from skimage.morphology import binary_closing
from skimage.transform import probabilistic_hough_line
from skimage import draw
from skimage.morphology import skeletonize
import time
from scipy import spatial
from rdp import rdp, pldist
import shapely.geometry as splyG
from shapely_geojson import dump, Feature, FeatureCollection
from scipy import ndimage as ndi
import os

from pipeline.map_object_classifier import MapObjectsClassifier
from pipeline.utils import cropImage, glueImage

epsilon = 4
min_poly_area = 1500
max_poly_area = 300000


def make_as_polygon(p1_big):
    if not p1_big.is_empty:
        if not p1_big.type == 'Polygon' and not p1_big.type == 'LineString':
            cur_area = 0
            for polygon in list(p1_big):
                if polygon.area > cur_area:
                    p1_big = polygon
                    cur_area = polygon.area
    return p1_big


def difference_with_enlarged_polygons(polygon_points, masked_contours, contours_extract=True):
    contour_features = splyG.Polygon(masked_contours).simplify(epsilon)
    for label, zip_and_pts in polygon_points.items():
        p1_big = polygon_points[label]['shapely_obj'].buffer(method_parameter, join_style=splyG.JOIN_STYLE.mitre)
        try:
            p1_big = p1_big.intersection(contour_features)
        except Exception:
            pass
        for zipper_label in zip_and_pts['neighboring_labels']:
            if zipper_label in polygon_points:
                p2 = polygon_points[zipper_label]['shapely_obj']
                if contours_extract: p2 = p2.buffer(method_parameter, join_style=splyG.JOIN_STYLE.mitre)
                if not p1_big.is_empty:
                    try:
                        p1_big = p1_big.difference(p2)
                    except Exception:
                        pass

        if not p1_big.is_empty:
            polygon_points[label]['shapely_obj'] = p1_big
            if contours_extract:
                try:
                    contour_features = contour_features.difference(p1_big)
                except Exception:
                    pass
            p1_big = make_as_polygon(p1_big)
            polygon_points[label]['points'] = p1_big.exterior.coords
    return polygon_points, contour_features


def validate_polygon(poly):
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.type == 'Polygon':
            cur_area = 0
            for polygon in list(poly):
                if polygon.area > cur_area:
                    poly = polygon
                    cur_area = polygon.area
                    if not poly.is_valid:
                        poly = poly.envelope
    return poly


def douglas_peucker(contours, douglas):
    global epsilon
    epsilon = douglas
    outer = splyG.Polygon(np.squeeze(contours[0])).simplify(epsilon)
    inners = []
    if hasattr(outer, 'interiors'):
        for hole in outer.interiors:
            inner = splyG.Polygon(hole).simplify(epsilon)
            inners.append(inner.exterior.coords)
    return validate_polygon(splyG.Polygon(outer.exterior.coords, inners))


def morphological_closing(labels, label):
    label_segmented = np.zeros(labels.shape, dtype="uint8")
    label_segmented[labels == label] = 255
    c_edges, top, bottom = cropImage(label_segmented)
    inner_region_closed = binary_closing(c_edges)
    neighboring_labels = labels[top[0]:bottom[0] + 20, top[1]:bottom[1] + 20]
    return glueImage(inner_region_closed, label_segmented, top, bottom), neighboring_labels


def get_candidate_point(a, b, point, threshhold, mid_point=False):
    dp = b - a
    st = dp[0] ** 2 + dp[1] ** 2

    # check if point inside the line
    u = (((point[0] - a[0]) * dp[0] + (point[1] - a[1]) * dp[1]) / st)

    if 0.05 <= u <= 0.95 and pldist(point, a, b) < threshhold:
        if mid_point:
            return [(point[0] + a[0] + (u * dp[0])) // 2, (point[1] + a[1] + (u * dp[1])) // 2]
        else:
            return [a[0] + (u * dp[0]), a[1] + (u * dp[1])]
    else:
        return None


def merge_lines_with_points(poly_1, poly_2, threshold=5.):
    aux_poly_2 = []
    aux_poly_1 = poly_1.copy()
    for idx2, a in enumerate(poly_2):
        aux_poly_2.append(a)
        b = poly_2[(idx2 + 1) % len(poly_2)]
        for idx1, p in enumerate(aux_poly_1):
            pt = get_candidate_point(a, b, p, threshold, mid_point=False)
            if pt:
                poly_1[idx1] = pt
                aux_poly_2.append(pt)
    return poly_1, np.array(aux_poly_2)


def merge_nearest_points_efficent(poly_1, poly_2, size=5):
    poly_tree = spatial.KDTree(np.vstack((poly_1, poly_2)))
    pairs_points = poly_tree.query_pairs(size)
    for a, b in pairs_points:
        if a < poly_1.shape[0] <= b:
            point_a = poly_1[a]
            point_b = poly_2[b - poly_1.shape[0]]
            poly_1[a] = (point_b + point_a) // 2
            poly_2[b - poly_1.shape[0]] = (point_b + point_a) // 2
        elif b < poly_1.shape[0] <= a:
            point_b = poly_1[b]
            point_a = poly_2[a - poly_1.shape[0]]
            poly_1[b] = (point_b + point_a) // 2
            poly_2[a - poly_1.shape[0]] = (point_b + point_a) // 2
    return poly_1, poly_2


def snapping_polygons(polygon_points):
    for label, zip_and_pts in polygon_points.items():
        poly_1 = np.array(list(polygon_points[label]['points']))
        for zipper_label in zip_and_pts['neighboring_labels']:
            poly_2 = np.array(list(polygon_points[zipper_label]['points']))
            poly_2, poly_1 = merge_lines_with_points(poly_2, poly_1, method_parameter)
            poly_1, poly_2 = merge_nearest_points_efficent(poly_1, poly_2, method_parameter)
            try:
                polygon_points[zipper_label]['neighboring_labels'].remove(label)
            except:
                pass
            polygon_points[label]['points'] = poly_1
            polygon_points[zipper_label]['points'] = poly_2
    return polygon_points, None


def post_processing(labels, color_classifier: MapObjectsClassifier, img, method, param, douglas, contours_extract=True):
    global method_parameter
    method_parameter = param

    # refine colors
    mask = labels.copy()
    mask[mask > 0] = 255
    img = cv2.medianBlur(img, 5)
    img = exposure.equalize_hist(img, mask=mask)

    contours_mask, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    labels_unique, cnt = np.unique(labels, return_counts=True)
    labels_filtered = [labels_unique[idx] for idx, l in enumerate(cnt.tolist()) if l > 700]

    polygon_points = dict()
    for label in reversed(labels_filtered):
        if label == 0:
            continue

        closing, neighboring_labels = morphological_closing(labels, label)

        find_polygons, _ = cv2.findContours(closing.astype(np.uint8), cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)

        poly = douglas_peucker(find_polygons, douglas)

        if not poly.is_empty:
            polygon_points[label] = dict()
            polygon_points[label]['neighboring_labels'] = [neighbor for neighbor in np.unique(neighboring_labels) if
                                                           neighbor in labels_filtered and
                                                           neighbor != 0 and
                                                           neighbor != label]
            polygon_points[label]['points'] = poly.exterior.coords
            polygon_points[label]['shapely_obj'] = poly
            polygon_points[label]['properties'] = color_classifier.predict(img, closing, poly, contours_mask)

    if method == 1:
        return difference_with_enlarged_polygons(polygon_points,
                                                 masked_contours=np.squeeze(
                                                     contours_mask[0]),
                                                 contours_extract=contours_extract)
    elif method == 2:
        return snapping_polygons(polygon_points)
