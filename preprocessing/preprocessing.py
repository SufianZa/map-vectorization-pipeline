import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageChops


def detectOuterContours(img):
    edges = cv2.Canny(img, 149, 150, 1)
    kernel = np.ones((7, 7), np.uint8)
    out = cv2.dilate(edges, kernel)
    # cv2.imwrite("TESTCANNY.jpg", edges)
    # plt.imshow(edges, cmap='gray'), plt.show()
    #
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # out = np.zeros_like(edges)
    # cv2.drawContours(out, contours, -1, 255, 1)
    # edges = cv2.dilate(out, kernel, iterations=2)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    # out = np.zeros_like(edges)
    # contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(out, contours, -1, 255, 1)
    return out


def overlayImages(img1, img2):
    plt.imshow(img2, cmap='gray')
    plt.imshow(img1, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()


def addPadding(image, pad):
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 255)
    image[image == 0] = 255
    return image


def contoursFiller(im):
    copy_im = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(copy_im, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(copy_im)
    im_out = im | im_floodfill_inv
    return im_out


def removeBorders(reg_im1, im2):
    mask = im2[:,:,2] < 250
    im2 = im2[np.ix_(mask.any(1), mask.any(0))]
    reg_im1 = reg_im1[np.ix_(mask.any(1), mask.any(0))]
    return reg_im1, im2


def register_images(im1, im2, dist='reg'):
    print("Aligning images ...", end=" ")
    w1, h1, _ = im1.shape
    w2, h2, _ = im2.shape

    # down sample map image if needed
    if w2 != w1 or h1 != h2:
        im1 = im1[::2, ::2]

    # remove edge distortion
    im1 = im1[30:, 30:][:w1 - 100, :h1 - 100]
    im2 = im2[30:, 30:][:w2 - 100, :h2 - 100]

    # add padding
    im1 = addPadding(im1, 500)
    im2 = addPadding(im2, 500)

    # convert to grayscale
    g_im1 = cv2.cvtColor(im1, cv2.COLOR_RGBA2GRAY)
    g_im2 = cv2.cvtColor(im2, cv2.COLOR_RGBA2GRAY)

    # detect outer contours
    contours_im1 = detectOuterContours(g_im1)
    contours_im2 = detectOuterContours(g_im2)


    # create masks of the outer contours
    filled_im1 = contoursFiller(contours_im1)
    filled_im2 = contoursFiller(contours_im2)
    filled_im2 = cv2.blur(filled_im2, (3, 3))


    # align images

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

    # Draw top matches
    imMatches = cv2.drawMatches(g_im1, keypoints1, g_im2, keypoints2, matches, None, flags=4)
    imMatches2 = cv2.drawMatches(filled_im1, keypoints1, filled_im2, keypoints2, matches, None, flags=4)
    # plt.imshow(imMatches2, cmap='gray'), plt.show()
    cv2.imwrite(dist + "/matches.jpg", imMatches)
    cv2.imwrite(dist + "/matches2.jpg", imMatches2)

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
    if w2 != w1 or h1 != h2:
        reg_im1 = np.ones_like(im2)*255
        reg_im1 = cv2.warpPerspective(im1, h, (width, height))
    else:
        reg_im1 = im1

    # trim borders
    reg_im1, im2 = removeBorders(reg_im1, im2)

    # ## show result
    # plt.imshow(overlay_im, cmap='gray'), \
    # plt.text(overlay_im.shape[0] // 2, overlay_im.shape[1] // 2, dist, horizontalalignment='center',
    #          verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5)), plt.show()
    # ################################################################

    print("Done.", flush=True)

    assert reg_im1.shape == im2.shape
    return reg_im1, im2

def generateTrainBatches(reg_im1, im2, batch_size=(400,400)):

    pass