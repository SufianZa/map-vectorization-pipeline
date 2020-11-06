import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature


def detectOuterContours(img):
    edges = cv2.Canny(img, 70, 70, 1)
    plt.imshow(edges), plt.show()
    kernel = np.ones((7, 7), np.uint8)
    out = cv2.dilate(np.array(edges).astype(np.uint8), kernel)
    edges = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(edges)
    cv2.drawContours(out, contours, -1, 255, 1)
    edges = cv2.dilate(out, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    out = np.zeros_like(edges)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, 255, 1)
    out = cv2.dilate(out, kernel, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=3)
    return out


def addPadding(image, pad):
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 255)
    result[result == 0] = 255
    return result


def contoursFiller(im):
    copy_im = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(copy_im, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(copy_im)
    im_out = im | im_floodfill_inv
    return im_out


def removeBorders(reg_im1, im2):
    mask = im2[:, :, 2] < 250
    im2 = im2[np.ix_(mask.any(1), mask.any(0))]
    reg_im1 = reg_im1[np.ix_(mask.any(1), mask.any(0))]
    return reg_im1, im2


def register_images(moving_image, fixed_image):
    """
    This method apply the image registration of tow similar images by using their silhouette to detect
    their most outer common features
    :param moving_image: ndarray
        the image to be transformed using affine transformation
    :param fixed_image: ndarray
        the image to be transformed using affine transformation
    """
    print("Aligning images ...", end=" ")
    w1, h1, _ = moving_image.shape
    w2, h2, _ = fixed_image.shape

    # down sample map image if needed
    if w1 > w2 * 2 or h1 > h2 * 2:
        moving_image = moving_image[::2, ::2]

    # remove edge distortion
    moving_image = moving_image[30:, 30:][:w1 - 100, :h1 - 100]
    fixed_image = fixed_image[30:, 30:][:w2 - 100, :h2 - 100]

    # add padding in order to recognise the feature points on the edge of the map
    moving_image = addPadding(moving_image, 500)
    fixed_image = addPadding(fixed_image, 500)

    # convert to grayscale
    g_im1 = cv2.cvtColor(moving_image, cv2.COLOR_RGBA2GRAY)
    g_im2 = cv2.cvtColor(fixed_image, cv2.COLOR_RGB2GRAY)

    # plt.imshow(filled_im2), plt.show()
    # detect outer contours
    contours_im1 = detectOuterContours(g_im1)
    contours_im2 = detectOuterContours(g_im2)

    # create binary masks of the outer contours
    filled_im1 = contoursFiller(contours_im1)
    filled_im2 = contoursFiller(contours_im2)
    filled_im2 = cv2.blur(filled_im2, (3, 3))

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(500)
    keypoints1, descriptors1 = orb.detectAndCompute(filled_im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(filled_im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.2)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = g_im2.shape
    reg_im1 = cv2.warpPerspective(moving_image, h, (width, height))

    # trim borders
    reg_im1, fixed_image = removeBorders(reg_im1, fixed_image)
    assert reg_im1.shape == fixed_image.shape
    return reg_im1, fixed_image
