import cv2
import numpy as np
import time
import random
import utils
from Tag import Tag
from Ocr import OCR
from config import params
from PIL import Image
import pytesseract as tess
from Video import Video


class beeTracker(object):
    def __init__(self):
        self.tags = []
        self.prevtags = [[]]

    def filter_tags(self, tags):
        """ By inspecting previous coordinates, return a filtered tag list. """

        def isFeasible(tag):
            for prevtags in self.prevtags:
                if not len(prevtags):
                    return True
                for t in prevtags:
                    if utils.dist_between(t.coords, tag.coords) < 15:
                        return True
                return False

        for tag in tags:
            if not isFeasible(tag):
                tags.remove(tag)
        return tags

    def label_bees(self, img):
        """ Return a frame with tagged bees labeled with their tag color. """
        self.tags = get_tags(img, tolerance=params["HoughTolerance"])
        # self.tags = self.filter_tags(tags)
        self.prevtags.append(self.tags[:])
        if len(self.prevtags) > 10:
            self.prevtags.pop(0)
        for tag in self.tags:
            color = tag.get_tag_color(img)
            if color != "UNSURE":
                # Backdrop for lulz
                cv2.putText(img, color, (int(tag.coords[0] + 52), int(tag.coords[1] - 23)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 0), thickness=2)
                # Display color of tag
                cv2.putText(img, color, (int(tag.coords[0] + 50), int(tag.coords[1] - 25)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 0), thickness=2)
                cv2.circle(img, tag.coords, tag.size, color=(0, 255, 255), thickness=4)

        # for t in self.tags:
        #     # Draw the circle
        #     cv2.circle(img, t.coords, t.size, color=(0, 255, 255), thickness=4)
        return img


def filter_coords(prevCoords, newcoord, tolerance):
    p = 3
    if len(prevCoords) >= p:
        avgp = 0
        for i in range(1, p + 1):
            avgp += np.array(prevCoords[-i]) / float(p)
        print "Average: {} New: {} Dist: {} Tol: {}".format(avgp, newcoord, utils.dist_between(avgp, newcoord),
                                                            tolerance)
        if utils.dist_between(avgp, newcoord) > tolerance:
            print "Rejected"
            return False
    print "Approved"
    return True


def get_tags(img, tolerance=20):
    """ Return a list of bee tags. Each tag has a coordinate. """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    avg_r = 35  # height * params["TagSizeScreenHeightRatio"]
    min_r = int(avg_r * params["TagSizeMinPercentage"])
    max_r = int(avg_r * params["TagSizeMaxPercentage"])
    # print height, width, avg_r, min_r, max_r
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                               minDist=params["MinTagDist"], minRadius=min_r, maxRadius=max_r, param2=tolerance)
    tags = []
    if len(circles[0] == 0):
        for circle in circles[0]:
            t = Tag()
            t.coords = circle[0], circle[1]
            t.size = circle[2]
            # print "SIZE:", t.size
            tags.append(t)
        return tags
    else:
        print "ERROR: No circles found in image"


def display_tags(img, tags):
    """ Display image with bee tags highlighted. """
    if tags:
        for t in tags:
            # Draw the circle
            cv2.circle(img, t.coords, t.size, color=(0, 255, 255), thickness=4)
    utils.display_img(img)


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


def realtime_playback(video):
    """ Plays a video with performing operations on each frame. Computations are real-time. """
    cap = cv2.VideoCapture(video)
    frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_ct):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Operations on the frame come here
        label_bees(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def offline_playback(video, fps=10):
    """ Plays a video with performing operations on each frame. Computes operations offline."""
    cap = cv2.VideoCapture(video)
    frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    processed = []

    for i in range(frame_ct):
        ret, frame = cap.read()
        frames.append(frame)

    tracker = beeTracker()
    for frame in frames:
        f = tracker.label_bees(frame)
        processed.append(f)

    raw_input("READY! Press Enter to continue...")
    for i in processed:
        cv2.imshow('frame', i)
        time.sleep(1 / fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return processed


def split_video(video):
    # FIXME: This is a mess, try with smaller video before going to 4K
    cap = cv2.VideoCapture(video)
    frame_ct = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    seq = []
    prev_coords = []
    tolerance = 30
    for i in range(frame_ct):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Operations on the frame come here
        tags = get_tags(frame)

        # FIXME: Still very messy, but not terrible. Clean this up.
        tag = None

        for t in tags:
            if get_tag_color(frame, t) == "YELLOW":

                # Display demo ###############
                # Candidate tag
                cv2.circle(frame, (t.coords[0], t.coords[1]), t.size, (255, 255, 255))
                # Search area

                p = 3
                avgp = (0, 0)
                if len(prev_coords) >= p:
                    for i in range(1, p + 1):
                        avgp += np.array(prev_coords[-i]) / float(p)

                cv2.circle(frame, (int(avgp[0]), int(avgp[1])), int(tolerance), (50, 50, 255))
                display_img(frame)
                ##############################

                if filter_coords(prev_coords, t.coords, tolerance):
                    tolerance = 30
                    prev_coords.append(t.coords)
                    tag = t
                else:
                    tolerance *= 1.25

        if tag:
            height, width, channels = frame.shape
            r = params["CropSize"]
            # Correct point of interest to be bounded
            # print height, width, r
            # print tag.coords[0], tag.coords[1]
            x, y = min(max(r, int(tag.coords[0])), width - r), min(max(r, int(tag.coords[1])), height - r)
            # print "Corrected: ", x, y
            # Crop tag area
            cropped = frame[y - r:y + r, x - r:x + r]
            seq.append(cropped)
    print len(seq)
    if seq:
        output_video(seq, name='../vid/output.avi')


def output_video(images, name='video.avi'):
    """ Given a list of images, create a video. """
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP42'), 30, (width, height))
    for image in images:
        # display_img(image)
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def get_nth_frame_video(video, n):
    """ Return nth frame from a video file."""
    cap = cv2.VideoCapture(video)
    frame = None
    if int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) < n:
        print "ERROR: n must be smaller than total frame count!"
    else:
        for i in range(n):
            ret, frame = cap.read()
    return frame


def tesseract_read(img, psm=8):
    """ Return text in given image as a string. For best results preprocess the input image. """
    # display_img(img)
    img = Image.fromarray(img)
    text = tess.image_to_string(img,
                                config="-c tessedit_char_whitelist=0123456789 load_system_dawg=false load_freq_dawg=false -psm %d" % psm)
    if text != "":
        print text
    return text


# TODO: OCR Accuracy
# TODO: Split video into individual bee chunks -- Kinda here

# ocr = OCR()
# ocr.train('../img/ocr_train.png')
# img = cv2.imread('../img/TestSet.png', 1)
# utils.display_img(img)
# tags = get_tags(img)
# for tag in tags:
#     tag.get_tag_id(img)



# split_video("../vid/videoDemo.mp4")

# # Load the image in color
# img = cv2.imread('../img/OneYellowBee.png', 1)
# offline_playback("../vid/videoDemo.mp4")
# output_video(offline_playback("../vid/videoDemo.mp4", fps=16.0))
#
#
# img = cv2.imread('../img/OneWhiteOneYellowBee.png', 1)
# img = label_bees(img)
# display_tags(img, get_tags(img))
# utils.display_img(img)

# display_img(img)
# tags = get_tags(img)


