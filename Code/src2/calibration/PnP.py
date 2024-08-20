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
fx, fy, ppx, ppy = 383.806, 383.806, 320.490, 240.654
cameraMatrix1 = np.array([
    [fx, 0,  ppx],
    [0,  fy, ppy],
    [0,  0,  1]
])
distCoeffs1 = np.array([0,0,0,0,0], dtype=np.float32)

# Camera 2 intrinsics : serial_number2 = "815412070846"
fx, fy, ppx, ppy = 382.799, 382.799, 317.411, 234.821
cameraMatrix2 = np.array([
    [fx, 0,  ppx],
    [0,  fy, ppy],
    [0,  0,  1]
])
distCoeffs2 = np.array([0,0,0,0,0], dtype=np.float32)

# Chessboard pattern size
pattern_size = (9, 6)
square_size = 6

# Chessboard points in object coordinate system
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# Load images
path1 = "back.png"
path2 = "front.png"
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# Find chessboard corners (findChessboardCornersSB gives better results than findChessboardCorners)
ret1, corners1 = cv2.findChessboardCornersSB(img1, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)
ret2, corners2 = cv2.findChessboardCornersSB(img2, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
# corners1 = cv2.cornerSubPix(img1, corners1, (10, 10), (-1,-1), criteria)
# corners2 = cv2.cornerSubPix(img2, corners2, (10, 10), (-1,-1), criteria)

# Compute the extrinsics parameters using solvePnP
_, rvec1, tvec1 = cv2.solvePnP(objp, corners1, cameraMatrix1, distCoeffs1)
_, rvec2, tvec2 = cv2.solvePnP(objp, corners2, cameraMatrix2, distCoeffs2)

# Visualize the chessboard corners and the axes
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
xyz_axes_img1, _ = cv2.projectPoints(axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
xyz_axes_img2, _ = cv2.projectPoints(axis, rvec2, tvec2, cameraMatrix2, distCoeffs2)
img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_copy = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img1_copy = cv2.drawChessboardCorners(img1_copy, pattern_size, corners1, ret1)
img2_copy = cv2.drawChessboardCorners(img2_copy, pattern_size, corners2, ret2)
img1_copy = draw_axes(img1_copy, corners1[0], xyz_axes_img1)
img2_copy = draw_axes(img2_copy, corners2[0], xyz_axes_img2)
cv2.imshow("Camera 1", img1_copy)
cv2.imshow("Camera 2", img2_copy)

# Compute the rotation matrices
R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)

# Compute the position of the cameras in object coordinate system
center1_pos = -R1.T @ tvec1
center2_pos = -R2.T @ tvec2
distance = np.linalg.norm(center1_pos - center2_pos)
distance_cm = distance * square_size
print("Distance between the two cameras: {} cm".format(distance_cm))

""" # Rotate the world coordinate system around the Z axis
# Theta angle in radians
theta = -10 * np.pi / 180

# Z axis rotation
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

new_axis = Rz @ axis.T
xyz_axes_img1, _ = cv2.projectPoints(new_axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
xyz_axes_img2, _ = cv2.projectPoints(new_axis, rvec2, tvec2, cameraMatrix2, distCoeffs2)
img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_copy = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img1_copy = cv2.drawChessboardCorners(img1_copy, pattern_size, corners1, ret1)
img2_copy = cv2.drawChessboardCorners(img2_copy, pattern_size, corners2, ret2)
img1_copy = draw_axes(img1_copy, corners1[0], xyz_axes_img1)
img2_copy = draw_axes(img2_copy, corners2[0], xyz_axes_img2)
cv2.imshow("Camera 1", img1_copy)
cv2.imshow("Camera 2", img2_copy)

new_center1_pos = Rz @ center1_pos
new_center2_pos = Rz @ center2_pos
print("Center 1 position in new system:", new_center1_pos)
print("Center 2 position in new system:", new_center2_pos) """

def get_euler_angles(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rotation_matrix)
    euler_angles_rad = r.as_euler('xyz', degrees=False)
    euler_angles_deg = np.degrees(euler_angles_rad)
    return euler_angles_deg

def get_transform_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def get_inverse_homogenous(R, t):
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -(R.T @ t).flatten()
    return T

euler_angles_deg1 = get_euler_angles(rvec1)
euler_angles_deg2 = get_euler_angles(rvec2)
print("Euler angles:", euler_angles_deg1)
print("Euler angles:", euler_angles_deg2)

# Compute the translation between camera 1 and camera 2
center1_in_W = -R1.T @ tvec1
center1_in_W_homogenous = np.hstack((center1_in_W.flatten(), 1))
world_to_cam2 = get_transform_matrix(R2, tvec2)
center1_in_cam2_homogenous = world_to_cam2 @ center1_in_W_homogenous
center1_in_cam2 = center1_in_cam2_homogenous[:3]
print("Center 1 in cam2 coordinate system:", center1_in_cam2)
distance_cm = np.linalg.norm(center1_in_cam2) * square_size
print("Distance between the two cameras in cam2 system:", distance_cm)

# Rotate around the x-axis to align the two cameras
y = abs(center1_in_cam2[1])
z = abs(center1_in_cam2[2])
theta = - np.arctan(y/z)
theta = -27 * np.pi / 180
print("Theta angle:", np.degrees(theta))
R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

new_center1_in_cam2 = R_x @ center1_in_cam2
camera_translation = new_center1_in_cam2.ravel() * square_size
camera_translation = [np.format_float_positional(x) for x in camera_translation]
print("Camera translation", camera_translation)

# Angles
cam1_angle = 90 - abs(euler_angles_deg1[0])
cam2_angle = 90 - abs(euler_angles_deg2[0])
print("Camera 1 angle:", cam1_angle)
print("Camera 2 angle:", cam2_angle)

cv2.waitKey(0)

W_to_C2 = get_transform_matrix(R2, tvec2)
C2_to_W = get_inverse_homogenous(R2, tvec2)
W_to_C1 = get_transform_matrix(R1, tvec1)
C1_to_W = get_inverse_homogenous(R1, tvec1)

C1_to_C2 = W_to_C2 @ C1_to_W
translation = C1_to_C2[:3, 3]
print("Translation between camera 1 and camera 2:", translation)




""" fig = plt.figure()
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
plt.show() """





