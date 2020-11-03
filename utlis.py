import cv2


def addPadding(image, top=0, bottom=0, left=0, right=0, a=0):
    image = cv2.copyMakeBorder(image.copy(), top + a, bottom + a, left + a, right + a, cv2.BORDER_CONSTANT, 255)
    image[image == 0] = 255
    return image


def removePadding(image, top=0, bottom=0, left=0, right=0, a=0):
    w, h = image.shape
    return image.copy()[left + a:w - bottom - a, top + a:h - right - a]
