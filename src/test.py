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
import time
import random
from Tag import Tag
from config import params


def get_tags(img, tolerance=20):
    """ Return a list of bee tags. Each tag has a coordinate. """
    height, width = img.shape
    avg_r = height / 20.0
    min_r = int(avg_r * params["TagSizeMinPercentage"])
    max_r = int(avg_r * params["TagSizeMaxPercentage"])
    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1,
                               minDist=params["MinTagDist"], minRadius=min_r, maxRadius=max_r, param2=tolerance)
    tags = []
    for circle in circles[0]:
        t = Tag()
        t.coords = circle[0], circle[1]
        t.size = circle[2]
        tags.append(t)
    return tags


def display_img(img, text='Image'):
    cv2.imshow(text, img)
    cv2.moveWindow('Image', params["WindowPosX"], params["WindowPosY"])
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
    """ Return True if a pixel is in given HSV range """
    h, s, v = hsv
    maxh, maxs, maxv = max_lim
    minh, mins, minv = min_lim
    if minh <= h <= maxh and mins <= s <= maxs and minv <= v <= maxv:
        return True
    else:
        return False


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


def in_frame_bounds(img, coords):
    """ Return True if given coordinates are withing the image. """
    height, width, channels = img.shape
    if coords[0] < height-1 and coords[1] < width-1:
        return True
    else:
        return False


def get_color_beliefs(img, tag):
    """ Return a dictionary of beliefs associated with a tag being white or yellow. """
    # Convert to HSV
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsvimg)
    # Tag parameters
    x = int(tag.coords[0])
    y = int(tag.coords[1])
    r = int(tag.size * 0.9)
    # Initialize accumulators
    acc_y = 0.0
    acc_w = 0.0

    # Check randomly sampled pixels in circle --- FAST
    samples = params["ColorDetectSamples"]
    for i in range(samples):
        i = random.randrange(-r, r)
        j = random.randrange(-r, r)

        if in_frame_bounds(img, (y+i, x+j)):
            hsv = h[y + i][x + j], s[y + i][x + j], v[y + i][x + j]
            # Test for yellow color
            if in_color_bounds(hsv, np.array(params["YellowMin"]), np.array(params["YellowMax"])):
                acc_y += 1
            # Test for white color
            elif in_color_bounds(hsv, np.array(params["WhiteMin"]), np.array(params["WhiteMax"])):
                acc_w += 1

    # Return percentages
    white = acc_w / samples
    yellow = acc_y / samples
    return {"WHITE": white, "YELLOW": yellow}


def get_tag_color(img, tag, threshold=0.5):
    """ Return tag color as string. Return "UNSURE" if all beliefs are under a threshold. """
    beliefs = get_color_beliefs(img, tag)
    best = max(beliefs.iterkeys(), key=lambda k: beliefs[k])
    if beliefs[best] < threshold:
        return "UNSURE"
    else:
        return best


def label_bees(img):
    """ Return a frame with tagged bees labeled with their tag color. """
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = get_tags(img_bw, tolerance=params["HoughTolerance"])
    for tag in tags:
        color = get_tag_color(img, tag, threshold=params["ColorThreshold"])
        if color != "UNSURE":
            # Backdrop for lulz
            cv2.putText(img, color, (int(tag.coords[0]+52), int(tag.coords[1]-23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 0), thickness=2)
            # Display color of tag
            cv2.putText(img, color, (int(tag.coords[0]+50), int(tag.coords[1]-25)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 0), thickness=2)
    return img


def realtime_playback(video):
    """ Plays a video with performing operations on each frame. Computations are real-time. """
    cap = cv2.VideoCapture(video)
    frame_ct = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    for i in range(frame_ct):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Operations on the frame come here
        label_bees(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def offline_playback(video):
    """ Plays a video with performing operations on each frame. Computes operations offline."""
    cap = cv2.VideoCapture(video)
    frame_ct = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frames = []
    processed = []

    for i in range(frame_ct):
        ret, frame = cap.read()
        frames.append(frame)

    for frame in frames:
        f = label_bees(frame)
        processed.append(f)

    raw_input("READY! Press Enter to continue...")
    for i in processed:
        cv2.imshow('frame', i)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# TODO: Split video into individual bee chunks
# TODO: Read bee tag OCR

if __name__ == "__main__":
    # # Load the image in color
    # img = cv2.imread('../img/OneYellowBee.png', 1)
    # offline_playback("../vid/videoDemo.mp4")
    realtime_playback("../vid/videoDemo.mp4")


