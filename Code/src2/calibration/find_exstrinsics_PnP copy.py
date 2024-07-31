import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy.spatial.transform import Rotation as R

# Blue : x-axis
# Green : y-axis
# Red : z-axis

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    #print("Corner : ", corner)
    #print("imgpts : ", imgpts)
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
square_size = 6

# Detect chessboard corners
path1 = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\1_3.png"
path2 = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\realsense_calibration_matrix\\test\\new\\2_3.png"

objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

ret1, corners1 = cv2.findChessboardCornersSB(img1, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)
ret2, corners2 = cv2.findChessboardCornersSB(img2, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)

_, rvec1, tvec1 = cv2.solvePnP(objp, corners1, cameraMatrix1, distCoeffs1)
_, rvec2, tvec2 = cv2.solvePnP(objp, corners2, cameraMatrix2, distCoeffs2)

imgpts1, _ = cv2.projectPoints(axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
imgpts2, _ = cv2.projectPoints(axis, rvec2, tvec2, cameraMatrix2, distCoeffs2)
img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_copy = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img1_copy = cv2.drawChessboardCorners(img1_copy, pattern_size, corners1, ret1)
img2_copy = cv2.drawChessboardCorners(img2_copy, pattern_size, corners2, ret2)
img1_copy = draw(img1_copy, corners1, imgpts1)
img2_copy = draw(img2_copy, corners2, imgpts2)
cv2.imshow("Camera 1", img1_copy)
cv2.imshow("Camera 2", img2_copy)

R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)

# center2_pos = -R2.T @ tvec2
# center1_pos = -R1.T @ tvec1
# print("Center 1 position:", center1_pos)
# print("Center 2 position:", center2_pos)
# distance1 = np.linalg.norm(center1_pos - center2_pos)
# print("Distance between the two cameras:", distance1)

# # Theta angle in radians
# theta = -3 * np.pi / 180

# # Z axis rotation
# Rz = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta), np.cos(theta), 0],
#     [0, 0, 1]
# ])

# new_axis = Rz @ axis.T
# imgpts1, _ = cv2.projectPoints(new_axis, rvec1, tvec1, cameraMatrix1, distCoeffs1)
# imgpts2, _ = cv2.projectPoints(new_axis, rvec2, tvec2, cameraMatrix2, distCoeffs2)
# img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
# img2_copy = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# img1_copy = cv2.drawChessboardCorners(img1_copy, pattern_size, corners1, ret1)
# img2_copy = cv2.drawChessboardCorners(img2_copy, pattern_size, corners2, ret2)
# img1_copy = draw(img1_copy, corners1, imgpts1)
# img2_copy = draw(img2_copy, corners2, imgpts2)
# cv2.imshow("Camera 1 new system", img1_copy)
# cv2.imshow("Camera 2 new system", img2_copy)

# new_center1_pos = Rz @ center1_pos
# new_center2_pos = Rz @ center2_pos
# print("Center 1 position in new system:", new_center1_pos)
# print("Center 2 position in new system:", new_center2_pos)
# distance2 = np.linalg.norm(new_center1_pos - new_center2_pos)
# print("Distance between the two cameras in new system:", distance2)

def get_euler_angles(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rotation_matrix)
    euler_angles_rad = r.as_euler('xyz', degrees=False)
    euler_angles_deg = np.degrees(euler_angles_rad)
    return euler_angles_deg

euler_angles_deg1 = get_euler_angles(rvec1)
euler_angles_deg2 = get_euler_angles(rvec2)
print("Euler angles:", euler_angles_deg1)
print("Euler angles:", euler_angles_deg2)


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

center1_in_W = -R1.T @ tvec1
center1_in_W_homogenous = np.hstack((center1_in_W.flatten(), 1))
world_to_cam2 = get_transform_matrix(R2, tvec2)
center1_in_cam2 = world_to_cam2 @ center1_in_W_homogenous
center1_in_cam2 = center1_in_cam2[:3]
print("Center 1 in cam2 coordinate system:", center1_in_cam2[:3])
distance3 = np.linalg.norm(center1_in_cam2)
print("Distance between the two cameras in cam2 system:", distance3)

y = abs(center1_in_cam2[1])
z = abs(center1_in_cam2[2])
theta = - np.arctan(y/z)
print("Theta angle:", np.degrees(theta))
R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

center2_in_cam2 = R_x @ np.array([0, 0, 0])
new_center1_in_cam2 = R_x @ center1_in_cam2
print("New center 1 in cam2 coordinate system:", new_center1_in_cam2[:3] * square_size)


# euler_angles_deg1 = get_euler_angles(rvec1)
# center1_pos = -R1.T @ tvec1
# print("-------------------")
# print("Camera 1")
# print("Euler angles:", euler_angles_deg1)q
# print("Translation vector:", tvec1.flatten() * square_size)
# print("Center position:", center1_pos.flatten())
# print("-------------------")

# euler_angles_deg2 = get_euler_angles(rvec2)
# center2_pos = -R2.T @ tvec2
# print("-------------------")
# print("Camera 2")
# print("Euler angles:", euler_angles_deg2)
# print("Translation vector:", tvec2.flatten() * square_size)
# print("Center position:", center2_pos.flatten())
# print("-------------------")

# def get_transform_matrix(R, t):
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = t.flatten()
#     return T

# def get_inverse_homogenous(R, t):
#     T = np.eye(4)
#     T[:3, :3] = R.T
#     T[:3, 3] = -(R.T @ t).flatten()
#     return T

# center2_in_cam2_coordinate_system = np.array([0, 0, 0, 1])
# center1_in_cam1_coordinate_system = np.array([0, 0, 0, 1])

# cam1_to_W = get_transform_matrix(R1, tvec1)
# W_to_cam1 = get_inverse_homogenous(R1, tvec1)
# origin_W = np.array([0, 0, 0, 1])
# origin_cam1 = W_to_cam1 @ origin_W
# origin_W_2 = cam1_to_W @ origin_cam1
# print("Origin world:", origin_W, "Origin world 2:", origin_W_2)
# print("translation vector cam1 to W (position of cam1 in W system): ", cam1_to_W[:3, 3])
# print("translation vector W to cam1: (position of W in cam1 system)", W_to_cam1[:3, 3])

# W_to_cam2 = get_inverse_homogenous(R2, tvec2)

# cam1_to_cam2 = W_to_cam2 @ cam1_to_W

# center1_in_cam2_coordinate_system = cam1_to_cam2 @ center1_in_cam1_coordinate_system

# print("Center 2 in cam2 coordinate system:", center2_in_cam2_coordinate_system[:3] * square_size)
# print("Center 1 in cam2 coordinate system:", center1_in_cam2_coordinate_system[:3] * square_size)




cv2.waitKey(0)
# # Matrices de transformation homogènes
# T1 = np.eye(4)
# T1[:3, :3] = R1
# T1[:3, 3] = tvec1.flatten()
# print("T1: ", T1)

# T2 = np.eye(4)
# T2[:3, :3] = R2
# T2[:3, 3] = tvec2.flatten()

# # Inversion de T1
# T1_inv = np.eye(4)
# T1_inv[:3, :3] = R1.T
# T1_inv[:3, 3] = -(R1.T @ tvec1).flatten()

# # Transformation de la caméra 1 vers la caméra 2
# T_2_to_1 = T2 @ T1_inv

# # Centre de la caméra 1 en coordonnées homogènes
# center1_hom = np.array([0, 0, 0, 1])

# # Calcul du centre de la caméra 2
# center2_transformed_hom = T_2_to_1 @ center1_hom
# center2_transformed = center2_transformed_hom[:3]

# # Calcul direct du centre de la caméra 2
# center1 = -R1.T @ tvec1
# center2 = -R2.T @ tvec2

# # Affichage des résultats
# print("Center camera 1:", center1.flatten())
# print("Center camera 2 (calcul direct):", center2.flatten())
# print("Centre de la caméra 1:", [0, 0, 0])
# print("Centre de la caméra 2 (transformation):", center2_transformed)