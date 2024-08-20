import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import utils

# Read angles values from rotation_90.txt
filename = 'calibration_results\\rotation_90.txt'
with open(filename) as file:
    lines = [line.rstrip() for line in file]

front_angle = float(lines[0])
side_angle = float(lines[1])

def crop_body(pcd, center, angle):
    rad = angle * np.pi / 180
    R = pcd.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    center_bbox_min = center - np.array([0.5, 0, 0.5])
    center_bbox_max = center + np.array([0.5, 0, 0.5])

    # Compute y_min as the first percentile in case the actual min_bound is an outlier
    y_values = np.asarray(pcd.points)[:, 1]
    y_min = np.percentile(y_values, 1)

    center_bbox_min[1] = y_min + 0.2 # 0.2 to remove the floor
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    return body

capture = False

if capture:
    front_pipe = utils.create_pipeline("815412070846")
    side_pipe = utils.create_pipeline("815412070753")

    # Wait 5 seconds
    for i in range(5):
        print("Capture will start in ", 5-i, " seconds")
        time.sleep(1)

    # Save the PLY files
    utils.save_ply("front.ply", front_pipe)
    utils.save_ply("side.ply", side_pipe)

front_path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras_90\\backpack_in_hands\\front.ply"
side_path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras_90\\backpack_in_hands\\side.ply"

# Load front point cloud
pcd_front = o3d.io.read_point_cloud(front_path)
pcd_front = pcd_front.voxel_down_sample(voxel_size=0.012)

# Load side point cloud
pcd_side = o3d.io.read_point_cloud(side_path)
pcd_side = pcd_side.voxel_down_sample(voxel_size=0.012)

# Crop the body
front_body_center = np.array([0, 0, -1.5]) # Center of the body : computed manually by pressing shift + left click in the visualization window
front_body = crop_body(pcd_front, front_body_center, front_angle)

side_body_center = np.array([0, 0, -1.2]) # Center of the body : computed manually by pressing shift + left click in the visualization window
side_body = crop_body(pcd_side, side_body_center, side_angle)

utils.visualize(front_body)
utils.visualize(side_body)

# Project the side body to the XZ plane
ground_projection_side = utils.project_to_xz_plane(side_body)
utils.visualize(ground_projection_side)

# Get bounding box
body_bbox_side = ground_projection_side.get_axis_aligned_bounding_box()
body_dimensions = body_bbox_side.get_max_bound() - body_bbox_side.get_min_bound()
print("Body dimensions side: ", body_dimensions)

# Project the front body to the XZ plane
ground_projection_front = utils.project_to_xz_plane(front_body)
utils.visualize(ground_projection_front)

# Get bounding box
body_bbox_front = ground_projection_front.get_axis_aligned_bounding_box()
body_dimensions = body_bbox_front.get_max_bound() - body_bbox_front.get_min_bound()
print("Body dimensions front: ", body_dimensions)
