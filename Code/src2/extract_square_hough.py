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

# 3. Find parallel lines
parallel_lines_angles = {}
parallel_lines_lengths = {}
seen = set()

for key1, angle1 in angles.items():
    for key2, angle2 in angles.items():
        if key1 == key2:
            continue

        if (key1, key2) in seen or (key2, key1) in seen:
            continue

        if abs(angle1 - angle2) < 5:
            if abs(lengths[key1] - lengths[key2]) < 10:
                distance = np.sqrt((lines_probabilistic[key1][0][0] - lines_probabilistic[key2][0][0]) ** 2 + (lines_probabilistic[key1][0][1] - lines_probabilistic[key2][0][1]) ** 2)
                if 100 < distance < 300:
                    parallel_lines_angles[(key1, key2)] = (angle1 + angle2)/2
                    parallel_lines_lengths[(key1, key2)] = (lengths[key1] + lengths[key2])/2
        
        seen.add((key1, key2))
        

# Remove outliers with Z-score method, based on angles
print(parallel_lines_angles)
angles_values = np.array(list(parallel_lines_angles.values()))
mean = np.mean(angles_values)
std = np.std(angles_values)
z_scores = (angles_values - mean) / std
threshold = 3
inliers_keys_angle = [key for key, z_score in zip(parallel_lines_angles.keys(), z_scores) if abs(z_score) < 3]
parallel_lines_angles = {key: value for key, value in parallel_lines_angles.items() if key in inliers_keys_angle}
print(parallel_lines_angles)

# Remove outliers with Z-score method, based on lengths
lengths_values = np.array(list(parallel_lines_lengths.values()))
mean = np.mean(lengths_values)
std = np.std(lengths_values)
z_scores = (lengths_values - mean) / std
threshold = 3
inliers_keys_length = [key for key, z_score in zip(parallel_lines_lengths.keys(), z_scores) if abs(z_score) < 3]

inliers_keys = set(inliers_keys_angle).intersection(set(inliers_keys_length))

# Draw parallel lines
image_parallel = image.copy()
image_parallel = cv2.cvtColor(image_parallel, cv2.COLOR_GRAY2BGR)

for key1, key2 in inliers_keys:
    x1, y1, x2, y2 = lines_probabilistic[key1][0]
    cv2.line(image_parallel, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1, y1, x2, y2 = lines_probabilistic[key2][0]
    cv2.line(image_parallel, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.namedWindow('Parallel lines', cv2.WINDOW_NORMAL)
cv2.imshow('Parallel lines', image_parallel)
cv2.resizeWindow('Parallel lines', 864, 648)
cv2.waitKey(0)
cv2.destroyAllWindows()
