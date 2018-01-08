import cv2
import numpy as np
import time
import random
from Tag import Tag
from config import params
from PIL import Image
import pytesseract as tess


def get_tags(img, tolerance=20):
    """ Return a list of bee tags. Each tag has a coordinate. """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    avg_r = height / 20.0
    min_r = int(avg_r * params["TagSizeMinPercentage"])
    max_r = int(avg_r * params["TagSizeMaxPercentage"])
    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1,
                               minDist=params["MinTagDist"], minRadius=min_r, maxRadius=max_r, param2=tolerance)
    tags = []
    if len(circles[0]==0):
        for circle in circles[0]:
            t = Tag()
            t.coords = circle[0], circle[1]
            t.size = circle[2]
            tags.append(t)
        return tags
    else:
        print "ERROR: No circles found in image"


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

    # Check randomly sampled pixels in circle --- FASTER than checking all pixels
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


def get_tag_color(img, tag, threshold=params["ColorThreshold"]):
    """ Return tag color as string. Return "UNSURE" if all beliefs are under a threshold. """
    beliefs = get_color_beliefs(img, tag)
    best = max(beliefs.iterkeys(), key=lambda k: beliefs[k])
    if beliefs[best] < threshold:
        return "UNSURE"
    else:
        return best


def label_bees(img):
    """ Return a frame with tagged bees labeled with their tag color. """
    tags = get_tags(img, tolerance=params["HoughTolerance"])
    for tag in tags:
        color = get_tag_color(img, tag)
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


def output_video(images, name='video.avi'):
    """ Given a list of images, create a video. """
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(name, cv2.cv.CV_FOURCC(*'MP42'), 30, (width, height))
    for image in images:
        # display_img(image)
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def clean_img(img):
    """ Return a cleaner version of image """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # display_img(img)
    ret, img = cv2.threshold(img, 185, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    # display_img(img)
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # display_img(img)
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # display_img(img)
    return img


def tesseract_read(img, psm=8):
    """ Return text in given image as a string. For best results preprocess the input image. """
    # display_img(img)
    img = Image.fromarray(img)
    text = tess.image_to_string(img, config="-c tessedit_char_whitelist=0123456789 load_system_dawg=false load_freq_dawg=false -psm %d" % psm)
    if text != "":
        print text
    return text


def crop_tag(img, tag):
    """ Return cropped tag image with clear background. """
    x, y, r = int(tag.coords[0]), int(tag.coords[1]), int(tag.size)
    # Crop tag area
    cropped_img = img[y-r:y+r, x-r:x+r]
    # Remove background
    for i in range(r*2):
        for j in range(r*2):
            if (i-r)**2 + (j-r)**2 > r**2 - 250:
                cropped_img[i, j] = [255, 255, 255]
    return cropped_img


def get_tag_id(img, tag, rotation=0):
    """ Return tag number """
    cropped_img = crop_tag(img, tag)
    # Clean cropped image
    cropped_img = clean_img(cropped_img)
    # Rotate image - Optional
    cropped_img = rotate_img(cropped_img, rotation)
    # Read string
    tag_id = tesseract_read(cropped_img)
    tag_id = tag_id.replace(" ", "")
    if tag_id == "" or len(tag_id.split('\n'))!=1:
        # print "ERROR: Couldn't detect tag number."
        return None
    else:
        return int(tag_id)


def rotate_img(img, angle):
    if angle == 0:
        return img
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))


def rotation_invariant_tag_detect(img, tag):
    """ Return tag number invariant of tag rotation """
    candidates = []
    for i in range(72):
        tag_id = get_tag_id(img, tag, rotation=i*5)
        if tag_id:
            candidates.append(tag_id)
    print candidates
    if candidates:
        return max(candidates, key=candidates.count)
    else:
        return None


def dist_between(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) ** 0.5


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


def filter_coords(prevCoords, newcoord, tolerance=100):
    """ By inspecting previous coordinates, approve or reject a new coordinate as feasible """
    p = 3
    if len(prevCoords) >= p:
        avgp = 0
        for i in range(1, p+1):
            avgp += np.array(prevCoords[-i])/float(p)
        print "Average: {} New: {} Dist: {} Tol: {}".format(avgp, newcoord, dist_between(avgp, newcoord), tolerance)
        if dist_between(avgp, newcoord) > tolerance:
            print "Rejected"
            return False
    print "Approved"
    return True


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
            x, y = min(max(r, int(tag.coords[0])), width-r), min(max(r, int(tag.coords[1])), height-r)
            # print "Corrected: ", x, y
            # Crop tag area
            cropped = frame[y-r:y+r, x-r:x+r]
            seq.append(cropped)
    print len(seq)
    if seq:
        output_video(seq, name='../vid/output.avi')


# TODO: OCR Accuracy
# TODO: Split video into individual bee chunks -- Kinda here

if __name__ == "__main__":
    # split_video("../vid/videoDemo.mp4")

#     print "GROUND: 49,49,49,26,30"
#     psm = 9
#     print "-------------------"
#     tesseract_read("../img/A.png",psm)
#     tesseract_read("../img/B.png",psm)
#     tesseract_read("../img/C.png",psm)
#     tesseract_read("../img/D.png",psm)
#     tesseract_read("../img/E.png",psm)
#     print "##"
#     tesseract_read("../img/A_rot.png",psm)
#     tesseract_read("../img/B_rot.png",psm)
#     tesseract_read("../img/C_rot.png",psm)
#     tesseract_read("../img/D_rot.png",psm)
#     tesseract_read("../img/E_rot.png",psm)

    # # Load the image in color
    # img = cv2.imread('../img/OneYellowBee.png', 1)
    offline_playback("../vid/videoDemo.mp4")
    # realtime_playback("../vid/C0007.MP4")


    # img = cv2.imread('../img/TwoWhiteBees.png', 1)

    # display_img(img)
    # tags = get_tags(img)

    # print rotation_invariant_tag_detect(img, tags[1])
    # print rotation_invariant_tag_detect(img, tags[2])

