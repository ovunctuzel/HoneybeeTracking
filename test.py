# from PIL import Image
# import pytesseract as tess
#
# im = Image.open("Bee.png")
#
# text = tess.image_to_string(im)
#
# print text

import cv2
import numpy as np
from Tag import Tag


def get_tags(img, tolerance=20):
    """ Return a list of bee tags. Each tag has a coordinate. """
    height, width = img.shape
    avg_r = height / 20.0
    min_r = int(avg_r * 0.8)
    max_r = int(avg_r * 1.2)
    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1,
                               minDist=10, minRadius=min_r, maxRadius=max_r, param2=tolerance)
    tags = []
    for circle in circles[0]:
        t = Tag()
        t.coords = circle[0], circle[1]
        t.size = circle[2]
        tags.append(t)
    return tags


def display_img(img, text='Image'):
    cv2.imshow(text, img)
    cv2.moveWindow('Image', 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_tags(img, tags):
    """ Display image with bee tags highlighted. """
    if tags:
        for t in tags:
            # Draw the circle
            cv2.circle(img, t.coords, t.size, color=(0, 255, 255), thickness=4)
    display_img(img)


def in_color_bounds(hsv, min_lim, max_lim):
    h, s, v = hsv
    maxh, maxs, maxv = max_lim
    minh, mins, minv = min_lim
    if h > maxh or h < minh or s > maxs or s < mins or v > maxv or v < minv:
        return False
    else:
        return True


def saturate(img, intensity=1.5):
    """ Saturate an image with a given intensity. """
    # Convert to HSV
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsvimg)
    # Multiply saturation
    s = s * intensity
    s = np.clip(s, 0, 255)
    s = s.astype(np.uint8)
    # Merge image back
    hsvimg = cv2.merge([h, s, v])
    # Back to BGR
    img = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
    return img


def get_color_beliefs(img, tag):
    """ Return a dictionary of beliefs associated with a tag being white or yellow. """
    # Convert to HSV
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsvimg)
    # Tag parameters
    x = int(tag.coords[0])
    y = int(tag.coords[1])
    r = int(tag.size)
    # Initialize accumulators
    acc_y = 0.0
    acc_w = 0.0
    total = (2 * r) ** 2
    for i in range(-r, r):
        for j in range(-r, r):
            hsv = h[y + i][x + j], s[y + i][x + j], v[y + i][x + j]
            # Test for yellow color
            if in_color_bounds(hsv, np.array([15, 30, 150]), np.array([50, 150, 255])):
                acc_y += 1
            # Test for white color
            if in_color_bounds(hsv, np.array([0, 0, 200]), np.array([180, 25, 255])):
                acc_w += 1
    # Return percentages
    yellow = acc_w / total
    white = acc_y / total
    return {"WHITE" : white, "YELLOW" : yellow}


def get_tag_color(img, tag, threshold=0.5):
    """ Return tag color as string. Return "UNSURE" if all beliefs are under a threshold. """
    beliefs = get_color_beliefs(img, tag)
    best = max(beliefs.iterkeys(), key=lambda k: beliefs[k])
    if best < threshold:
        return "UNSURE"
    else:
        return best

def label_bees(img):
    """ Display a frame with tagged bees labeled with their tag color. """
    pass


if __name__ == "__main__":
    # Load the image in black & white
    img_bw = cv2.imread('img/OneWhiteOneYellowBee.png', 0)
    # Load the image in color
    img = cv2.imread('img/OneWhiteOneYellowBee.png', 1)
    tags = get_tags(img_bw)
    # display_tags(img, tags)
    # img = saturate(img, 1.5)
    print get_tag_color(img, tags[0])
    print get_tag_color(img, tags[1])
