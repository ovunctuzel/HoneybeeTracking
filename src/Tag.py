import cv2
from config import params
from Ocr import OCR
import utils
import random
import numpy as np

class Tag(object):
    def __init__(self):
        self.coords = None
        self.size = None
        self.tag = None

    def get_color_beliefs(self, img):
        """ Return a dictionary of beliefs associated with a tag being white or yellow. """
        # Convert to HSV
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(hsvimg)
        # Tag parameters
        x = int(self.coords[0])
        y = int(self.coords[1])
        r = int(self.size * 0.9)
        # Initialize accumulators
        acc_y = 0.0
        acc_w = 0.0

        # Check randomly sampled pixels in circle --- FASTER than checking all pixels
        samples = params["ColorDetectSamples"]
        for i in range(samples):
            i = random.randrange(-r, r)
            j = random.randrange(-r, r)

            if utils.in_frame_bounds(img, (y+i, x+j)):
                hsv = h[y + i][x + j], s[y + i][x + j], v[y + i][x + j]
                # Test for yellow color
                if utils.in_color_bounds(hsv, np.array(params["YellowMin"]), np.array(params["YellowMax"])):
                    acc_y += 1
                # Test for white color
                elif utils.in_color_bounds(hsv, np.array(params["WhiteMin"]), np.array(params["WhiteMax"])):
                    acc_w += 1

        # Return percentages
        white = acc_w / samples
        yellow = acc_y / samples
        return {"WHITE": white, "YELLOW": yellow}

    def get_tag_color(self, img, threshold=params["ColorThreshold"]):
        """ Return tag color as string. Return "UNSURE" if all beliefs are under a threshold. """
        beliefs = self.get_color_beliefs(img)
        best = max(beliefs.iterkeys(), key=lambda k: beliefs[k])
        if beliefs[best] < threshold:
            return "UNSURE"
        else:
            return best

    def get_tag_id(self, img):
        """ Return tag number """
        cropped_img = utils.crop_tag(img, self)
        clean_img = utils.clean_img(cropped_img, tagsize=self.size)
        ocr = OCR()
        ocr.read_img(clean_img, tagsize=self.size)

        #
        # # Clean cropped image
        # cropped_img = utils.clean_img(cropped_img)
        # # Rotate image - Optional
        # cropped_img = utils.rotate_img(cropped_img, rotation)
        # # Read string
        # tag_id = tesseract_read(cropped_img)
        # tag_id = tag_id.replace(" ", "")
        # if tag_id == "" or len(tag_id.split('\n'))!=1:
        #     # print "ERROR: Couldn't detect tag number."
        #     return None
        # else:
        #     return int(tag_id)

