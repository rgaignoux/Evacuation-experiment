import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy.spatial.transform import Rotation as R

# Blue : x-axis
# Green : y-axis
# Red : z-axis

def draw_axes(img, origin, xyz_axes):
    origin = tuple(origin.ravel().astype(int))
    x = tuple(xyz_axes[0].ravel().astype(int))
    y = tuple(xyz_axes[1].ravel().astype(int))
    z = tuple(xyz_axes[2].ravel().astype(int))
    img = cv2.line(img, origin, x, (255,0,0), 5)
    img = cv2.line(img, origin, y, (0,255,0), 5)
    img = cv2.line(img, origin, z, (0,0,255), 5)
    return img


# Camera 1 intrinsics : serial_number1 = "815412070753"
fx, fy, ppx, ppy = 423.786, 423.786, 424.541, 240.722
cameraMatrix1 = np.array([
    [fx, 0,  ppx],
    [0,  fy, ppy],
    [0,  0,  1]
])
distCoeffs1 = np.array([0,0,0,0,0], dtype=np.float32)

# Chessboard pattern size
pattern_size = (9, 6)
square_size = 6

# Chessboard points in object coordinate system
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Load images
path1 = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\angles_test\\33_deg.png"
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

# Find chessboard corners (findChessboardCornersSB gives better results than findChessboardCorners)
ret1, corners1 = cv2.findChessboardCornersSB(img1, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)

# Compute the extrinsics parameters using solvePnP
_, rvec1, tvec1 = cv2.solvePnP(objp, corners1, cameraMatrix1, distCoeffs1)

# Visualize the chessboard corners and the axes
xyz_axes_img1, _ = cv2.projectPoints(axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img1_copy = cv2.drawChessboardCorners(img1_copy, pattern_size, corners1, ret1)
img1_copy = draw_axes(img1_copy, corners1[0], xyz_axes_img1)
cv2.imshow("Camera 1", img1_copy)

# Compute the rotation matrices
R1, _ = cv2.Rodrigues(rvec1)

# Compute the position of the cameras in object coordinate system
center1_pos = -R1.T @ tvec1

def get_euler_angles(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rotation_matrix)
    euler_angles_rad = r.as_euler('xyz', degrees=False)
    euler_angles_deg = np.degrees(euler_angles_rad)
    return euler_angles_deg

euler_angles_deg1 = get_euler_angles(rvec1)
print("Euler angles:", euler_angles_deg1)

cv2.waitKey(0)





