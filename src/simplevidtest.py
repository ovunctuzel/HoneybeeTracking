import cv2

images = []
for i in range(100):
    images.append(cv2.imread('1.png'))
    images.append(cv2.imread('2.png'))
    images.append(cv2.imread('3.png'))


height, width, layers = images[0].shape

video = cv2.VideoWriter('video.avi', cv2.cv.CV_FOURCC(*'MP42'), 30, (width, height))

for i in images:
    video.write(i)

cv2.destroyAllWindows()
video.release()