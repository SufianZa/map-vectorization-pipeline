import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import skeletonize, thin

from skimage.morphology import binary_closing


def watershed_transform(image, distance_transform, sensitivity):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray.copy()
    mask[gray < 253] = 255
    mask[gray > 253] = 0
    m_im = distance_transform.copy()
    m_im[distance_transform >= sensitivity] = 1
    m_im[distance_transform < sensitivity] = 0
    m_im = cv2.erode(1 - m_im, np.ones((5, 5), np.uint8), iterations=1)
    m_im[m_im < 1] = 0
    local_maxi = peak_local_max(distance_transform, indices=False,
                                footprint=np.ones((3, 3)), labels=m_im)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(1 - distance_transform, markers, mask=mask)
    return labels
