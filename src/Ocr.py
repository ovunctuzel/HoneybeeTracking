import sys
import numpy as np
import cv2
from config import params
import utils


class OCR(object):
    def __init__(self):
        self.model = cv2.KNearest()

    @staticmethod
    def train(imgpath):
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        samples = np.empty((0, params["FeatureVectorSize"][0] * params["FeatureVectorSize"][1]), np.float32)
        responses = []
        keys = [i for i in range(48, 58)]

        # TODO: Sort Contours

        for cnt in contours:
            if cv2.contourArea(cnt) >= params["TrainingDigitContourArea"]:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h >= params["TrainingDigitHeight"]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, params["FeatureVectorSize"])

                    cv2.imshow('norm', img)
                    key = cv2.waitKey(0) % 256

                    if key == 27:  # (escape to quit)
                        sys.exit()
                    elif key in keys:
                        responses.append(int(chr(key)))
                        sample = roismall.reshape((1, params["FeatureVectorSize"][0] * params["FeatureVectorSize"][1]))
                        samples = np.append(samples, sample, 0)

        responses = np.array(responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        print "Training complete..."

        samples = np.float32(samples)
        responses = np.float32(responses)

        cv2.imwrite("../data/train_result.png", img)
        np.savetxt('../data/generalsamples.data', samples)
        np.savetxt('../data/generalresponses.data', responses)

    def read_img(self, img, tagsize=params["AverageTagRadius"]):
        # ######    TRAINING    ############## #
        samples = np.loadtxt('../data/generalsamples.data', np.float32)
        responses = np.loadtxt('../data/generalresponses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        self.model.train(samples, responses)

        # ######    TESTING     ############## #
        out = np.zeros(img.shape, np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

        utils.display_img(img)
        utils.display_img(thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            print w, h, "vs", tagsize*1.25, tagsize*0.85, tagsize*0.2, tagsize
            if tagsize > w > tagsize*0.2 and tagsize*1.25 > h > tagsize*0.85:
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, params["FeatureVectorSize"])
                roismall = roismall.reshape((1, params["FeatureVectorSize"][0] * params["FeatureVectorSize"][1]))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k=params["kNearest"])
                string = str(int((results[0][0])))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

                imcp = img.copy()
                cv2.rectangle(imcp, (x, y), (x + w, y + h), (10, 255, 10), 2)
                utils.display_img(imcp)

        cv2.imshow('im', img)
        cv2.imshow('out', out)
        cv2.waitKey(0)

# ocr = OCR()
# img = cv2.imread('../img/TESTSETROTT.png')
# ocr.read_img(img)