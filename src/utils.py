import cv2
from config import params
import numpy as np


def display_img(img, text='Image'):
    cv2.imshow(text, img)
    cv2.moveWindow('Image', params["WindowPosX"], params["WindowPosY"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dist_between(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5


def in_frame_bounds(img, coords):
    """ Return True if given coordinates are within the image. """
    height, width, channels = img.shape
    if coords[0] < height - 1 and coords[1] < width - 1:
        return True
    else:
        return False


def in_color_bounds(hsv, min_lim, max_lim):
    """ Return True if a pixel is in given HSV range """
    h, s, v = hsv
    maxh, maxs, maxv = max_lim
    minh, mins, minv = min_lim
    if minh <= h <= maxh and mins <= s <= maxs and minv <= v <= maxv:
        return True
    else:
        return False


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# def clean_img(img):
#     """ Return a cleaner version of image """
#     display_img(img)
#     hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsvimg, (0, 0, 0), (180, 35, 255))
#     #
#     print "MASK"
#     display_img(mask)
#     print "MASK"
#     # mask = cv2.erode(mask, None, iterations=2)
#     # mask = cv2.dilate(mask, None, iterations=2)
#
#
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # img = cv2.GaussianBlur(img, (5, 5), 0)
#     # display_img(img)
#     # ret, img = cv2.threshold(img, 185, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
#     # display_img(img)
#     # kernel = np.ones((3, 3), np.uint8)
#     # img = cv2.dilate(img, kernel, iterations=1)
#     # display_img(img)
#     # kernel = np.ones((2, 2), np.uint8)
#     # img = cv2.erode(img, kernel, iterations=1)
#     # display_img(img)
#     return img


def crop_tag(img, tag):
    """ Return cropped tag image with clear background. """
    x, y, r = int(tag.coords[0]), int(tag.coords[1]), int(tag.size)
    # Crop tag area
    r = int(r * params["TagSizeCropFactor"])
    cropped_img = img[y - r:y + r, x - r:x + r]
    # Remove background
    # for i in range(r*2):
    #     for j in range(r*2):
    #         if (i-r)**2 + (j-r)**2 > r**2 - 250:
    #             cropped_img[i, j] = [255, 255, 255]
    return cropped_img


def remove_bg(img):
    """ Keep only the tag circle. """
    r = int(img.shape[0] / 2)
    for i in range(r * 2):
        for j in range(r * 2):
            if (i - r) ** 2 + (j - r) ** 2 > r ** 2 - 250:
                img[i, j] = 0
    return img


def rotate_img(img, angle):
    if angle == 0:
        return img
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows), borderValue=(0, 0, 0))


def clean_img(img, tagsize=params["AverageTagRadius"]):
    """ Given an image with a number, fix the rotation, threshold, and reduce noise. """
    # Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Remove BG
    img = remove_bg(thresh)
    # Remove speckles
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    # Invert colors
    # cv2.bitwise_not(img, img)
    # Fix the rotation
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        [w, h] = dist_between(box[0], box[1]), dist_between(box[1], box[2])
        if w * h >= tagsize * tagsize / 4:
            angle = rect[2]
            # Show contours
            box = np.int0(box)
            imcp = img.copy()
            cv2.drawContours(imcp, [box], 0, (0, 0, 255), 2)
            display_img(imcp)
            # Correct angle - Height should be larger than width
            if dist_between(box[0], box[1]) < dist_between(box[1], box[2]):
                angle += 90
            img = rotate_img(img, angle)
            break
    cv2.bitwise_not(img, img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
