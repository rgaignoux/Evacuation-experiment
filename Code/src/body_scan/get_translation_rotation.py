import numpy as np
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import utils

def get_distance_angles_180(front_intrinsics, back_intrinsics, pattern_size, square_size, path_IR_front, path_IR_back):
    # Front camera intrinsics
    cameraMatrix_front, distCoeffs_front = utils.get_instrinsics_matrix(front_intrinsics)

    # Front camera extrinsics
    rvec_front, tvec_front = utils.get_extrinsics(path_IR_front, pattern_size, cameraMatrix_front, distCoeffs_front)
    R_front, _ = cv2.Rodrigues(rvec_front)

    # Back camera intrinsics
    cameraMatrix_back, distCoeffs_back = utils.get_instrinsics_matrix(back_intrinsics)

    # Back camera extrinsics
    rvec_back, tvec_back = utils.get_extrinsics(path_IR_back, pattern_size, cameraMatrix_back, distCoeffs_back)
    R_back, _ = cv2.Rodrigues(rvec_back)

    # Compute the relative transformation between the cameras
    C_back_to_W = utils.get_inverse_homogenous(R_back, tvec_back)
    W_to_C_front = utils.get_transform_matrix(R_front, tvec_front)
    C_back_to_C_front = W_to_C_front @ C_back_to_W

    # Compute the cameras tilt angles on x-axis in degrees
    angle_front = utils.get_euler_angles(rvec_front)
    angle_back = utils.get_euler_angles(rvec_back)
    adjacent_angle_front = utils.get_adjacent_angle_90(angle_front[0])
    adjacent_angle_back = utils.get_adjacent_angle_90(angle_back[0])

    # Correct camera rotation to obtain the "flat" translation
    theta = adjacent_angle_front
    theta_rad = theta * np.pi / 180
    R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0, np.sin(theta_rad), np.cos(theta_rad)]
        ])

    translation = C_back_to_C_front[:3, 3] * square_size
    print("Translation before correction:", translation)
    corrected_translation = R_x @ translation
    print("Translation after correction:", corrected_translation)

    return corrected_translation, adjacent_angle_front, adjacent_angle_back


def get_angles_90(front_intrinsics, side_intrinsics, pattern_size, path_IR_front, path_IR_side):
    # Front camera intrinsics
    cameraMatrix_front, distCoeffs_front = utils.get_instrinsics_matrix(front_intrinsics)

    # Front camera extrinsics
    rvec_front, _ = utils.get_extrinsics(path_IR_front, pattern_size[::-1], cameraMatrix_front, distCoeffs_front)

    # Side camera intrinsics
    cameraMatrix_side, distCoeffs_side = utils.get_instrinsics_matrix(side_intrinsics)

    # Side camera extrinsics
    rvec_side, _ = utils.get_extrinsics(path_IR_side, pattern_size, cameraMatrix_side, distCoeffs_side)

    # Compute the cameras tilt angles on x-axis in degrees
    angle_front = utils.get_euler_angles(rvec_front)
    angle_side = utils.get_euler_angles(rvec_side)
    print("Front angle from extrinsics: ", angle_front)
    print("Side angle from extrinsics: ", angle_side)
    adjacent_angle_front = -(90 - abs(angle_front[0]))
    adjacent_angle_side = -(90 - abs(angle_side[0]))

    return adjacent_angle_front, adjacent_angle_side

# Camera parameters
width = 640
height = 480
fps = 15

# Chessboard parameters
pattern_size = (9, 6)
square_size = 6

##############
# 180° setup #
##############
setup_180 = False

if setup_180:
    # Create pipelines
    front_pipe, front_intrinsics = utils.create_pipeline_IR("815412070846", width, height, fps)
    back_pipe, back_intrinsics = utils.create_pipeline_IR("815412070753", width, height, fps)

    # Save IR images
    utils.save_IR(front_pipe, "front.png")
    utils.save_IR(back_pipe, "back.png")

    # Get distance and angles
    translation, angle_front, angle_back = get_distance_angles_180(front_intrinsics, back_intrinsics, pattern_size, square_size, "front.png", "back.png")

    np.set_printoptions(suppress=True,precision=3)
    print("Translation between the cameras:", translation)
    print("Front angle from extrinsics: ", angle_front)
    print("Back angle from extrinsics: ", angle_back)

    # Write those values to a file
    with open("translation_rotation_180.txt", "w") as f:
        f.write(str(translation[0]/100) + ' ' + str(translation[1]/100) + ' ' + str(translation[2]/100) + '\n')
        f.write(str(angle_front) + '\n')
        f.write(str(angle_back) + '\n')

    # Delete IR images
    os.remove("front.png")
    os.remove("back.png")

#############
# 90° setup #
#############
setup_90 = True

if setup_90:
    # Create pipelines
    front_pipe, front_intrinsics = utils.create_pipeline_IR("815412070846", width, height, fps)
    side_pipe, side_intrinsics = utils.create_pipeline_IR("815412070753", width, height, fps)

    # Save IR images
    utils.save_IR(front_pipe, "front.png")
    utils.save_IR(side_pipe, "side.png")

    # Get angles
    angle_front, angle_side = get_angles_90(front_intrinsics, side_intrinsics, pattern_size, "front.png", "side.png")

    np.set_printoptions(suppress=True,precision=3)
    print("Front angle from extrinsics: ", angle_front)
    print("Side angle from extrinsics: ", angle_side)

    # Write those values to a file
    with open("rotation_90.txt", "w") as f:
        f.write(str(angle_front) + '\n')
        f.write(str(angle_side) + '\n')

    # Delete IR images
    os.remove("front.png")
    os.remove("side.png")
