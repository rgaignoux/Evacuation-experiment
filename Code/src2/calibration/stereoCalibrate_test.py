import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

# Camera 1 intrinsics : serial_number1 = "815412070753"
fx, fy, ppx, ppy = 423.786, 423.786, 424.541, 240.722
cameraMatrix1 = np.array([
    [fx, 0,  ppx],
    [0,  fy, ppy],
    [0,  0,  1]
])
distCoeffs1 = np.array([0,0,0,0,0], dtype=np.float32)

# Camera 2 intrinsics : serial_number2 = "815412070846"
fx, fy, ppx, ppy = 422.674, 422.674, 421.141, 234.282
cameraMatrix2 = np.array([
    [fx, 0,  ppx],
    [0,  fy, ppy],
    [0,  0,  1]
])
distCoeffs2 = np.array([0,0,0,0,0], dtype=np.float32)

PATTERN_SIZE = (9, 6)
left_imgs = list(sorted(glob.glob("C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\1_*.png")))
right_imgs = list(sorted(glob.glob("C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\2_*.png")))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
left_pts, right_pts = [], []
img_size = None

for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if img_size is None:
        img_size = (left_img.shape[1], left_img.shape[0])
    
    res_left, corners_left = cv2.findChessboardCornersSB(left_img, PATTERN_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    res_right, corners_right = cv2.findChessboardCornersSB(right_img, PATTERN_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

    print(res_left, res_right)
    
    if res_left and res_right:
        #corners_left = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1,-1), criteria)
        #corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1,-1), criteria)

        # Draw and display the corners
        img_left = cv2.drawChessboardCorners(left_img, PATTERN_SIZE, corners_left, res_left)
        img_right = cv2.drawChessboardCorners(right_img, PATTERN_SIZE, corners_right, res_right)
        cv2.imshow('left', img_left)
        cv2.imshow('right', img_right)
        cv2.waitKey(0)
        
        left_pts.append(corners_left)
        right_pts.append(corners_right)

pattern_point = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_point[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points = [pattern_point] * len(left_pts)

err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(pattern_points, left_pts, right_pts, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img_size)

print('Left camera:')
print(Kl)
print('Left camera distortion:')
print(Dl)
print('Right camera:')
print(Kr)
print('Right camera distortion:')
print(Dr)
print('Rotation matrix:')
print(R)
print('Translation:')
print(T)

T_rel = np.hstack((R, T))
T_rel = np.vstack((T_rel, [0, 0, 0, 1]))

translation = T_rel[:3, 3]
square_size = 6
print("Translation :", translation * square_size)
distance = np.linalg.norm(translation)
print("Distance between cameras: ", distance * square_size)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

center1 = [0, 0, 0, 1]
center2 = np.dot(T_rel, center1)
ax.scatter(center1[0], center1[1], center1[2], c='r', label='Camera 1')
ax.scatter(center2[0], center2[1], center2[2], c='b', label='Camera 2')

# Draw the objps
for i in range(len(pattern_point)):
    point = pattern_point[i].reshape(-1)
    point1 = np.dot(T_rel, np.hstack((point, 1)))
    ax.scatter(point1[0], point1[1], point1[2], c='r')

ax.text(center1[0], center1[1], center1[2], 'Camera 1', color='red')
ax.text(center2[0], center2[1], center2[2], 'Camera 2', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.show()

