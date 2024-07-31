import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    print("Corner : ", corner)
    print("imgpts : ", imgpts)
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

# Chessboard pattern size
pattern_size = (9, 6)

# Detect chessboard corners
path1 = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\1_1.png"
path2 = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\2_1.png"

pts1 = []
pts2 = []

objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

ret1, corners1 = cv2.findChessboardCornersSB(img1, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)
ret2, corners2 = cv2.findChessboardCornersSB(img2, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)

print("ret1: ", ret1, "ret2: ", ret2)

if ret1 and ret2:
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    # corners1 = cv2.cornerSubPix(img1, corners1, (10, 10), (-1,-1), criteria)
    # corners2 = cv2.cornerSubPix(img2, corners2, (10, 10), (-1,-1), criteria)

    _, rvec1, tvec1 = cv2.solvePnP(objp, corners1, cameraMatrix1, distCoeffs1)
    _, rvec2, tvec2 = cv2.solvePnP(objp, corners2, cameraMatrix2, distCoeffs2)
    imgpts1, _ = cv2.projectPoints(axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
    imgpts2, _ = cv2.projectPoints(axis, rvec2, tvec2, cameraMatrix2, distCoeffs2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img1 = cv2.drawChessboardCorners(img1, pattern_size, corners1, ret1)
    img2 = cv2.drawChessboardCorners(img2, pattern_size, corners2, ret2)
    img1 = draw(img1, corners1, imgpts1)
    img2 = draw(img2, corners2, imgpts2)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

    pts1.append(corners1)
    pts2.append(corners2)

# Solve PnP
objps = [objp] * len(pts1)

pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)
objps = np.array(objps, dtype=np.float32)

# reshape to   Nx3 1-channel
objps = objps.reshape(-1, 1, 3)
# reshape to   Nx2 1-channel
pts1 = pts1.reshape(-1, 1, 2)
# reshape to   Nx2 1-channel
pts2 = pts2.reshape(-1, 1, 2)

_, rvec1, tvec1 = cv2.solvePnP(objps, pts1, cameraMatrix1, distCoeffs1)
_, rvec2, tvec2 = cv2.solvePnP(objps, pts2, cameraMatrix2, distCoeffs2)

print("rvec1: ", rvec1)
print("rvec2: ", rvec2)

# Compute relative transformation matrix between camera 1 and camera 2
R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)

_1TW = np.hstack((R1, tvec1)) # World to Camera 1
_1TW = np.vstack((_1TW, [0, 0, 0, 1]))

_2TW = np.hstack((R2, tvec2)) # World to Camera 2
_2TW = np.vstack((_2TW, [0, 0, 0, 1]))

print("Transformation matrix T1:\n", _1TW)
print("Transformation matrix T2:\n", _2TW)

# _2T1 = _2TW * _WT1

# Inverse of transformation matrix
_WT1 = np.eye(4)
_WT1[:3, :3] = R1.T
_WT1[:3, 3] = -R1.T @ tvec1.ravel()

_2T1 = _2TW @ _WT1
print("Transformation matrix T_rel:\n", _2T1)

square_size = 6

# Translation and distance between cameras
translation = _2T1[:3, 3]
print("Translation :", translation * square_size)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

center1 = np.array([0, 0, 0, 1])
center2 = _2T1 @ center1

print("Center 1: ", center1)
print("Center 2: ", center2)

center1 = center1[:3]
center2 = center2[:3]
center1 = center1.ravel()
center2 = center2.ravel()

distance = np.linalg.norm(center1 - center2)
distance = distance * square_size
print("Distance between cameras: ", distance)

print("Center 1: ", center1)
print("Center 2: ", center2)
ax.scatter(center1[0], center1[1], center1[2], c='r', label='Camera 1')
ax.scatter(center2[0], center2[1], center2[2], c='b', label='Camera 2')

ax.text(center1[0], center1[1], center1[2], 'Camera 1', color='red')
ax.text(center2[0], center2[1], center2[2], 'Camera 2', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.show()


# print("Translation :", translation * square_size)
# distance = np.linalg.norm(translation)
# print("Distance between cameras: ", distance * square_size)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

center1 = np.transpose(-R1) @ tvec1
center2 = np.transpose(-R2) @ tvec2
center1 = center1.ravel()
center2 = center2.ravel()

distance = np.linalg.norm(center1 - center2)
distance = distance * square_size
print("Distance between cameras: ", distance)

print("Center 1: ", center1)
print("Center 2: ", center2)
ax.scatter(center1[0], center1[1], center1[2], c='r', label='Camera 1')
ax.scatter(center2[0], center2[1], center2[2], c='b', label='Camera 2')

ax.text(center1[0], center1[1], center1[2], 'Camera 1', color='red')
ax.text(center2[0], center2[1], center2[2], 'Camera 2', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.show()