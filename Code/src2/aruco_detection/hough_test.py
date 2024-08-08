import cv2
import numpy as np

path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\images\\color_images\\images\\color_frame_811.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 1. Extract white pixels
white = cv2.inRange(image, 200, 255)
edges = cv2.Canny(white, 50, 150, apertureSize=3)

cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.imshow('Edges', edges)
cv2.resizeWindow('Edges', 864, 648)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Apply Hough transform
angles = {}
lengths = {}

image_hough = image.copy()
image_hough = cv2.cvtColor(image_hough, cv2.COLOR_GRAY2BGR)

lines_probabilistic = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
if lines_probabilistic is not None:
    count = 0
    for x1, y1, x2, y2 in lines_probabilistic[:, 0]:
        angle = np.arctan((y2 - y1) / (x1 - x2)) * 180 / np.pi
        angles[count] = angle

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lengths[count] = length

        count += 1
        cv2.line(image_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.namedWindow('Probabilistic Hough Line Transform', cv2.WINDOW_NORMAL)
cv2.imshow('Probabilistic Hough Line Transform', image_hough)
cv2.resizeWindow('Probabilistic Hough Line Transform', 864, 648)
cv2.waitKey(0)
cv2.destroyAllWindows()