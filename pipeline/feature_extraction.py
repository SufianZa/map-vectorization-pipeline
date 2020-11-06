from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from scipy.stats import skew
import pickle as pkl
import shapely.geometry as splyG
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2hsv
import os
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import make_pipeline
import global_variables
import enum
from pipeline.utils import mad, RGB2HEX


class FeaturesType(enum.Enum):
    """
    Enum class to define types of features and extract a set of all or selected features
    """
    SPECTRAL = 1
    SHAPE = 2
    SPECTRAL_SHAPE = 3

    def get_all_feature_list(self):
        if self == FeaturesType.SPECTRAL_SHAPE:
            return global_variables.cherry_pick_features[
                list(global_variables.spectral_index) + list(global_variables.shape_index)]
        elif self == FeaturesType.SPECTRAL:
            return global_variables.cherry_pick_features[global_variables.spectral_index]
        elif self == FeaturesType.SHAPE:
            return global_variables.cherry_pick_features[global_variables.shape_index]

    def get_selected_features(self):
        if self == FeaturesType.SPECTRAL_SHAPE:
            return global_variables.cherry_pick_features[
                global_variables.selected_spectral_features + global_variables.selected_shape_features]
        elif self == FeaturesType.SPECTRAL:
            return global_variables.cherry_pick_features[global_variables.selected_spectral_features]
        elif self == FeaturesType.SHAPE:
            return global_variables.cherry_pick_features[global_variables.selected_shape_features]

    def get_names_list(self, selected):
        return global_variables.spectral_shape_features_name[selected]


class FeatureExtractor:
    """
    FeatureExtractor extracts spectral and shape features from images and polygons
    """

    def __init__(self, selected_features, path, intensity_threshold=0.3):
        """
        The initialization consists of loading or extracting features
        :param selected_features: array
            the indices selected features
        :param path: str
            the path to load/save features
        :param intensity_threshold: float
           threshold to discard pixels with lower or equal intensity
        """
        self.INTENSITY_THRESHOLD = intensity_threshold
        self.selected_features = selected_features
        self.path = path
        try:
            self.x, self.y = self.load_features()
        except:
            print('Error or no feature file was found under {}'.format(path))
            self.extract_features()
            self.x, self.y = self.load_features()

    def __call__(self, *args, **kwargs):
        return self.get_selected_features(self.selected_features), self.y

    def extract_features(self):
        """
           Extracting features and labels from train images then the results are
           saved in the weight directory
       """
        print('Extracting features ...')
        x = []
        y = []
        for file_path in global_variables.train_full_maps.glob('*.*'):
            filename = str(os.path.basename(file_path))
            map_name = filename.split('.')[0]
            o_im = np.array(Image.open(file_path))[:, :, :3]
            o_im = cv2.medianBlur(o_im, 5)
            o_im = (o_im - np.min(o_im)) * 1.0 / (np.max(o_im) - np.min(o_im))
            gray = np.mean(o_im, axis=2)
            mask = gray.copy()
            mask[gray <= 0.99] = 1
            mask[gray > 0.99] = 0
            map_contours, _ = cv2.findContours(np.array(mask, dtype=np.uint8()) * 255, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

            # histogram equalization
            o_im = exposure.equalize_hist(o_im, mask=mask)

            for i, target_class in enumerate(global_variables.target_classes):
                print(target_class)
                # exclude contour class
                if i != 0:
                    try:
                        class_mask = np.array(Image.open(
                            os.path.join(global_variables.classes_path, '{}-{}.png'.format(map_name, target_class))))[:,
                                     :, 0]
                    except:
                        continue
                    contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 700:
                            # draw object as white mask
                            s_im = np.zeros_like(class_mask)
                            cv2.drawContours(s_im, [cnt], -1, 255, -1)

                            # convert as polygon
                            poly = splyG.Polygon(np.squeeze(cnt)).simplify(3)

                            # indices of object without background
                            ones = np.argwhere(np.logical_and(mask, s_im))

                            # image to 1x3-Vector
                            reshape = o_im[ones[:, 0], ones[:, 1], :]

                            # convert HSV
                            reshape = rgb2hsv(reshape)

                            # remove low intensity pixels
                            reshape = reshape[reshape[:, 2] > self.INTENSITY_THRESHOLD]
                            if reshape.shape[0] // 3 == 0:
                                continue
                            feature_vector = np.array(list(get_spectral_features(reshape)) + list(
                                get_shape_features(poly, map_contours[0])))

                            x.append(feature_vector)
                            y.append(target_class)
            break
        with open(self.path, "wb") as f:
            pkl.dump({'x': x, 'y': y}, f)

    def get_selected_features(self, select_features):
        """
        get a subset of selected features
        :param select_features: list
            list of indices of selected features
        :return ndarray
            the selected features of all data points
        """
        selected_x = []
        for feature in self.x:
            selected_x.append(feature[select_features])
        return selected_x

    def load_features(self):
        """
          Loads features from the weight directory
        """
        print('Loading features...')
        with open(self.path, "rb") as f:
            data = pkl.load(f)
        x = data['x']
        y = data['y']
        return x, y

    def feature_importance_permutation(self):
        """
            Calculate feature importance depending on the gini impurtiy of the random forest model
            the illustrates the results as graphs
         """
        print('Calculating feature importance and permutation importance...')
        for feature in FeaturesType:
            # select features
            selected_x = self.get_selected_features(feature.get_all_feature_list())
            x, x_test, y, y_test = train_test_split(selected_x, self.y, test_size=0.35, random_state=42,
                                                    stratify=self.y)

            # initilze random forest
            scaler = preprocessing.StandardScaler().fit(x)
            model = RandomForestClassifier(class_weight='balanced', max_depth=8, n_estimators=300)
            model.fit(scaler.transform(x), y)

            # get feature importance
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            # extract names of the features
            features_name = feature.get_names_list(feature.get_all_feature_list())

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle('Feature Importance analysis for {} features'.format(feature.name))

            # Plot the impurity-based feature importances of the forest
            plt.figure()
            ax2.set_title("Feature importances")
            ax2.bar(range(len(x[0])), importances[indices], color="r", yerr=std[indices], align="center")

            result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            ind_names = [features_name[i] for i in sorted_idx]
            ax1.boxplot(result.importances[sorted_idx].T,
                        vert=False, labels=ind_names)

            ax1.set_title("Permutation Importances (test set)")
            result = permutation_importance(model, x, y, n_repeats=10,
                                            random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            ind_names = [features_name[i] for i in sorted_idx]

            ax3.boxplot(result.importances[sorted_idx].T,
                        vert=False, labels=ind_names)

            ax3.set_title("Permutation Importances (train set)")
            plt.tight_layout()
            plt.show()

    def extract_features_from_polygon(self, image, poly_mask, poly, map_contours):
        """
        Extracts selected shape and spectral features from polygon to predict

        :param image: ndarray
                RGB input map image
        :param poly_mask: ndarray
                binary mask  of the polygon
        :param poly: shapely polyon object
                binary mask  of the polygon
        :param  map_contours: list of map outer contour points
        :return feature vector representation of a polygon
        """
        # cut slice from image and exclude black pixels
        ones = np.argwhere(poly_mask)
        reshape = image[ones[:, 0], ones[:, 1], :]

        # convert to hsv
        reshape = rgb2hsv(reshape)

        # remove low intensity pixels
        reshape = reshape[reshape[:, 2] > self.INTENSITY_THRESHOLD]
        if reshape.shape[0] < 20:
            return np.zeros_like(self.selected_features)

        feature_vector = np.array(list(get_spectral_features(reshape)) + list(
            get_shape_features(poly, map_contours[0])))

        return feature_vector[self.selected_features]


def visualize_features(x, y):
    """
    Use PCA dimensionality reduction to visualize the features in 2d space

    :param x: ndarray
           array of n-d feature vectors
    :param y: array
            labels
    """
    targets = list(global_variables.visualize_colors.keys())
    print(targets)
    colors = map(lambda color: RGB2HEX(color), list(global_variables.visualize_colors.values()))
    x_train = np.array(x)
    y_train = np.array(y)

    std_clf = make_pipeline(StandardScaler(), PCA(n_components=3), GaussianNB())
    std_clf.fit(x_train, y_train)
    scaler = std_clf.named_steps['standardscaler']
    pca_std = std_clf.named_steps['pca']
    principalComponents = pca_std.transform(scaler.transform(x_train))
    principalDf = pd.DataFrame(data=principalComponents[:, :2]
                               , columns=['principal component 1', 'principal component 2'])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PCA 1', fontsize=15)
    ax.set_ylabel('PCA 2', fontsize=15)
    finalDf = pd.concat([principalDf, pd.DataFrame(y_train)], axis=1)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[0] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)

    plt.grid()
    plt.legend(fontsize=20)
    plt.show()


def get_spectral_features(reshape):
    """
        Extracts spectral features from image vector including:
        - median
        - median absolute deviation
        - mean
        - standard deviation
        - skewness

    :param reshape: ndarray
        image vector in HSV space
    :return 16-d feature descriptor
     """
    return np.hstack((np.median(reshape, axis=0), mad(reshape, axis=0), np.mean(reshape, axis=0),
                      skew(reshape, axis=0), np.std(reshape, axis=0)))


def get_shape_features(polygon, map_contours):
    """
       Extracts shape features from polygons:
       - rectangularity
       - solidity
       - aspect_ratio
       - elongation
       - convexity
       - relative_area

        :param polygon: shapely polygon object
        :param map_contours: list of map outer contour points

        :return 6-d feature descriptor
    """
    map_poly = splyG.Polygon(np.squeeze(map_contours)).simplify(3)
    rect = polygon.minimum_rotated_rectangle
    x, y = rect.exterior.coords.xy
    edge_length = (splyG.Point(x[0], y[0]).distance(splyG.Point(x[1], y[1])),
                   splyG.Point(x[1], y[1]).distance(splyG.Point(x[2], y[2])))
    length = max(edge_length)
    breadth = min(edge_length)

    elongation = np.abs(length - breadth) / length
    relative_area = polygon.area / map_poly.area
    circularity = polygon.area / (polygon.length * polygon.length)
    convexity = (polygon.convex_hull.length / polygon.length)
    solidity = polygon.area / polygon.convex_hull.area
    aspect_ratio = length / breadth
    rectangularity = polygon.area / polygon.minimum_rotated_rectangle.area
    return rectangularity, solidity, aspect_ratio, elongation, circularity, convexity, relative_area
