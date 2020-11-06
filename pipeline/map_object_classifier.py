from pathlib import Path

import cv2
from skimage.color import rgb2hsv
from skimage.exposure import exposure
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import global_variables
import enum
from pipeline.feature_extraction import FeaturesType, FeatureExtractor


class Models(enum.Enum):
    RANDOMFOREST = 1
    SVM = 2


class MapObjectsClassifier:
    def __init__(self, features=FeaturesType.SPECTRAL_SHAPE, model=Models.RANDOMFOREST):
        self.featuresType = features
        self.extracted_features_path = Path(global_variables.weights_path, 'features.pkl')
        self.selected_features_list = np.array(features.get_selected_features())
        self.featureExtractor = FeatureExtractor(self.selected_features_list, self.extracted_features_path)
        x, y = self.featureExtractor()
        self.initialize_model(model, x, y, analysis=False)
        print('Map Objects Classifier using {} and {} features is Ready'.format(model.name, features.name))

    def predict(self, image, poly_mask, poly, map_contours):
        feature_vector = self.featureExtractor.extract_features_from_polygon(image, poly_mask, poly,
                                                                             map_contours)

        label = self.clf.predict(self.scaler.transform(feature_vector.reshape(1, -1)))[0]
        probability = self.clf.predict_proba(self.scaler.transform(feature_vector.reshape(1, -1)))[0]

        return dict(fill=global_variables.visualization_colors[label],
                    type=label,
                    probability=dict(background=probability[0], building=probability[1], water=probability[2]))

    def initialize_model(self, model, x, y, analysis=False):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        self.scaler = preprocessing.StandardScaler().fit(x_train)
        if model == Models.RANDOMFOREST:
            self.clf = RandomForestClassifier(class_weight='balanced', max_depth=8, n_estimators=300)
        elif model == Models.SVM:
            self.clf = SVC(probability=True)

        self.clf.fit(self.scaler.transform(x_train), y_train)

        if analysis:
            self.evaluation_print(self.scaler.transform(x_test), y_test, self.featuresType)

    def evaluation_print(self, x_test, y_test, featureType):
        y_pred = self.clf.predict(x_test)
        print(classification_report(y_test, y_pred))
        # Get and reshape confusion matrix data
        matrix = confusion_matrix(y_test, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=2.2)
        sns.heatmap(matrix, annot=True, annot_kws={'size': 25},
                    cmap=plt.cm.Reds)
        # Add labels to the plot
        class_names = ['background', 'building', 'water']
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.28
        tick_marks2[0] = tick_marks2[0] - 0.2
        tick_marks = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks2, class_names, rotation=90)
        plt.xlabel('Predicted label', labelpad=13)
        plt.ylabel('True label', labelpad=13)

        plt.title('Map Objects Classifier using {}'.format(featureType.name))
        # plt.savefig("conf-{}.pdf".format(featureType.name), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    MapObjectsClassifier(None, FeaturesType.SPECTRAL_SHAPE, Models.RANDOMFOREST)
