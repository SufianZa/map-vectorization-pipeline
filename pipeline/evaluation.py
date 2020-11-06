from PIL import Image
from scipy import ndimage as ndi
from tabulate import tabulate
import pandas as pd
from skimage.morphology import binary_erosion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import global_variables
from pipeline.run_pipeline import Pipeline
from pipeline.utils import crop_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class Evaluation:
    def __init__(self, thresholds=np.arange(0.5, 0.96, 0.05)):
        self.pred_num = 0
        self.true_num = 0
        self.results = pd.DataFrame(columns=['iou', 'pred_class', 'true_class', 'true_area', 'pred_area'])
        self.thresholds = thresholds

    def load_ground_truth(self, file_name, mask):
        """
        :param file_name: str
            the name of the map to get the ground truth image
        :param mask: ndarray
            binary image of the map without background
        :return: ndarray
            ground_truth ndarray of ground truth polygons as mask without contours
                labeled_classes ndarray of the classes labels after assigning each class to a number
        """
        ground_truth = np.zeros_like(mask, dtype=np.uint8)
        labeled_classes = np.zeros_like(mask, dtype=np.uint8)
        for i, target_class in enumerate(global_variables.target_classes):
            if i != 0:  # exclude contour class
                try:
                    class_mask = np.array(Image.open(
                        os.path.join(global_variables.classes_path,
                                     '{}-{}.png'.format(file_name, target_class))))[:, :, 0]
                except:
                    continue
                labeled_classes[class_mask.astype(bool)] = i
                ground_truth = np.logical_or(ground_truth, class_mask)
        ground_truth = np.logical_and(ground_truth, mask)
        return ground_truth.astype(np.uint8) * 255, labeled_classes

    def evaluate(self, polygon_points, path, mask, min_a=1500, max_a=300000):
        """
        :param polygon_points: a dictionary of all polygons extracted including the keys:
                "points"                simple representation of sequence of points
                "shapely_obj"           polygons as shapely objects
                "neighboring_labels"    list of the labels of the neighboring polygons
                "properties"            the classification of Map Object Classifier
        :param path: str or path
            the path of full input map image
        :param mask: ndarray
            binary image of the map without background
        :param min_a: int or float
            minimum area to include in the evaluation
        :param max_a: int or float
            maximum area to include in the evaluation
        """
        filename = os.path.basename(path)
        map_name = filename.split('.')[0]
        ground_truth_mask, labeled_classes = self.load_ground_truth(map_name, mask)
        gt_labels, _ = ndi.label(ground_truth_mask, structure=np.ones((3, 3)))
        gt_labels_unique, count_pixels = np.unique(gt_labels, return_counts=True)
        filtered_gt = [gt_labels_unique[idx] for idx, l in enumerate(count_pixels.tolist()) if max_a > l > min_a]
        self.true_num += len(filtered_gt)
        for poly_p in polygon_points.values():
            true_img, true_label, segment_id, pred_img, pred_label = self.matching_paris(gt_labels, poly_p,
                                                                                         labeled_classes)
            pred_area = len(np.argwhere(pred_img))
            true_area = len(np.argwhere(true_img))
            if segment_id not in filtered_gt or true_label == 0 or min_a > pred_area or pred_area > max_a:
                continue
            self.pred_num += 1
            entry = dict(iou=self.IoU(true_img, pred_img, show=True)
                         , pred_class=pred_label, true_class=global_variables.target_classes[true_label],
                         true_area=true_area, pred_area=pred_area)

            # append to evaluation dataFrame
            self.results.loc[len(self.results)] = entry

    def IoU(self, true_img, pred_img, show=False):
        """
            Calculates the Intersection over Union using binary mask images and logical operations
        :param true_img: ndarray
            the ground truth polygon as mask
        :param pred_img: ndarray
            the predicted polygon as mask
        :param show: bool
            to show overlaying polygon in 3*1 Figure
        :return: float
            IoU score
        """
        intersection = np.logical_and(true_img, pred_img)
        union = np.logical_or(true_img, pred_img)
        iou_score = np.sum(intersection) / np.sum(union)
        if show:
            color_true_red = np.zeros((*true_img.shape, 3))
            color_pred_blue = np.zeros((*pred_img.shape, 3))
            color_true_red[:, :, 0] = true_img
            color_pred_blue[:, :, 2] = pred_img
            overlay = cv2.addWeighted(color_true_red, 0.7, color_pred_blue, 0.3, 0)
            c_overlay, _, _ = crop_image(overlay)
            c_im1, _, _ = crop_image(color_true_red)
            c_im2, _, _ = crop_image(color_pred_blue)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(c_im1)
            ax1.title.set_text('Ground Truth')
            ax2.imshow(c_im2)
            ax2.title.set_text('Approximated')
            ax3.imshow(c_overlay)
            ax3.title.set_text('Overlay (IoU {:.4f})'.format(iou_score))
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            plt.show()
        return iou_score

    def geometrical_evaluation(self):
        """
        calculates the  'Precision', 'Recall', 'F1' and True Positives for different pre-defined thresholds (see self.thresholds)
        - plot a figure of histogram distribution of IoU scores
        - plot a figure of the correlation between the predicted area and its IoU scores
        """
        print('#Aproximated Polygons {}'.format(self.pred_num))
        print('#Ground Truth Polygons {}'.format(self.true_num))
        data = []
        for t in self.thresholds:
            tp = len(self.results[(self.results['iou'] >= t)])
            precision = float('{:0.4}'.format(tp / self.pred_num))
            recall = float('{:0.4}'.format(tp / self.true_num))
            f1 = 2 * (precision * recall) / (precision + recall)
            data.append(['t={:.2f}'.format(t), tp, precision, recall, f1])
        results = tabulate(data, headers=['Iou', 'Correct', 'Precision', 'Recall', 'F1'], floatfmt='.2f')

        # print info
        print(results)
        print('Average IoU scores {}'.format(np.array(self.results['iou']).mean()))

        # Visualize IoU Scores
        bars, bins = np.histogram(self.results['iou'], [0.0, 0.5, 0.8, 1.0])
        legend = ['{:.2f} - {:.2f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        print(legend)
        print(bars / bars.sum() * 100)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.hist(self.results['iou'], bins=100, color='black')
        plt.ylabel('#Polygons')
        plt.xlabel('IoU score')
        plt.locator_params(axis='x', nbins=15)
        plt.grid(True)
        plt.subplot(122)
        plt.bar(legend, bars / bars.sum() * 100)
        plt.grid(True)
        plt.xticks(rotation=0, fontsize=11)
        plt.ylabel('percentage %')
        plt.xlabel('IoU Interval')
        # plt.savefig("statsIou.pdf", bbox_inches='tight')
        plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        ax.scatter(self.results['iou'], self.results['pred_area'], s=0.5)
        ax.set_xlabel('IoU')
        ax.set_ylabel('Area in Pixels')
        plt.show()

    def map_objects_classifier_evaluation(self):
        """
        calculates the  'Precision', 'Recall', 'F1' and support for different classes to evaluate Map Objects
        Classifier
            - plot the confusion matrix of individual classes
            - prints a classification report
        """
        df = self.results[(self.results['iou'] > 0.7)]
        y_true = df['true_class']
        y_pred = df['pred_class']
        print(classification_report(y_true, y_pred))
        matrix = confusion_matrix(y_true, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        import seaborn as sns

        plt.figure(figsize=(10, 7))
        sns.set(font_scale=2.4)
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
        plt.show()

    def matching_paris(self, gt_labels, poly_p, classes_img):
        """
            Calculates the centroids of the predicted polygon to find the ground truth polygon in ground truth image
        :param gt_labels: includes all segments/polygons in ground truth image as a unique number
        :param poly_p: shapely object
        :param classes_img: includes all classes in ground truth image as a unique number
        :return:
            true_img: ndarray
                the ground truth polygon as mask
            pred_img: ndarray
                the predicted polygon as mask
            segment_id: int
                the assigned number of ground truth segment in gt_labels
            true_label: str
                the class of the ground truth polygon
            pred_label: str
                the class of the predicted polygon
        """
        true_img = np.zeros_like(gt_labels, dtype=np.uint8)
        pred_img = np.zeros_like(gt_labels, dtype=np.uint8)
        center_pt = poly_p['shapely_obj'].representative_point()
        segment_id = gt_labels[int(center_pt.y), int(center_pt.x)]
        true_label = classes_img[int(center_pt.y), int(center_pt.x)]
        true_img[gt_labels == segment_id] = 255
        cv2.drawContours(pred_img, np.array([poly_p['points']]).astype(int), -1, 255, -1)
        if hasattr(poly_p['shapely_obj'], 'interiors'):
            for i in poly_p['shapely_obj'].interiors:
                cv2.drawContours(pred_img, np.array([np.array(i)]).astype(int), -1, 0, -1)
        pred_img = binary_erosion(pred_img).astype(int) * 255
        return true_img, true_label, segment_id, pred_img, poly_p['properties']['type']


if __name__ == '__main__':
    evaluator = Evaluation()
    pipeline = Pipeline()
    for file_path in global_variables.test_full_maps.glob('*.*'):
        pipeline.run_pipeline(file_path, file_path, evaluator=evaluator)
    evaluator.geometrical_evaluation()
    evaluator.map_objects_classifier_evaluation()
