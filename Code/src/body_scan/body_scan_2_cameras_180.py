import numpy as np
import open3d as o3d
import numpy as np
import time
import utils

# Read translation & angles values from translation_rotation_180.txt
filename = 'calibration_results\\translation_rotation_180.txt'
with open(filename) as file:
    lines = [line.rstrip() for line in file]

translation = lines[0].split()
front_angle = float(lines[1])
back_angle = float(lines[2])

def crop_body(pcd_front, pcd_back, body_center, remove_outliers=True):
    # Rotate front pcd by 180 degrees on Y axis
    R = pcd_back.get_rotation_matrix_from_xyz((0, np.pi, 0))
    pcd_back.rotate(R, center=(0, 0, 0))

    # Rotate both pcds by the tilt angles on X axis
    rad = front_angle * np.pi / 180
    R = pcd_front.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd_front.rotate(R, center=(0, 0, 0))

    rad = back_angle * np.pi / 180
    R = pcd_back.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd_back.rotate(R, center=(0, 0, 0))

    # Translate
    pcd_front.translate(translation)

    # Merge both meshes
    pcd = pcd_front + pcd_back

    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()
    center_bbox_min = body_center - np.array([0.5, 0, 0.5])
    center_bbox_max = body_center + np.array([0.5, 0, 0.5])

    # Compute y_min as the first percentile in case the actual min_bound is an outlier
    y_values = np.asarray(pcd.points)[:, 1]
    y_min = np.percentile(y_values, 1)

    center_bbox_min[1] = y_min + 0.25 # 0.25 to remove the floor
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body using the bounding box
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    if remove_outliers:
        # Remove outliers
        cl, ind = body.remove_radius_outlier(nb_points=5, radius=0.02)
        body = body.select_by_index(ind)

    return body

capture = False

if(capture):
    back_pipe = utils.create_pipeline("815412070753")
    front_pipe = utils.create_pipeline("815412070846")

    # Wait 5 seconds
    for i in range(5):
        print("Capture will start in ", 5-i, " seconds")
        time.sleep(1)

    # Save the PLY files
    utils.save_ply("front.ply", front_pipe)
    utils.save_ply("back.ply", back_pipe)

front_path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras_180\\straight\\front.ply"
back_path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras_180\\straight\\back.ply"
# Load front point cloud
pcd_front = o3d.io.read_point_cloud(front_path)
pcd_front = pcd_front.voxel_down_sample(voxel_size=0.012)

# Load back point clouds
pcd_back = o3d.io.read_point_cloud(back_path)
pcd_back = pcd_back.voxel_down_sample(voxel_size=0.012)

# Remove outliers and visualize
cl, ind = pcd_front.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_front = pcd_front.select_by_index(ind)

cl, ind = pcd_back.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_back = pcd_back.select_by_index(ind)

# Get the body only
body_center = np.array([0, 0, 1.2]) # Center of the body : computed manually by pressing shift + left click in the visualization window
body = crop_body(pcd_front, pcd_back, body_center, remove_outliers=True)
utils.visualize(body)
# o3d.io.write_point_cloud("body.ply", body)

# Project the body to the XZ plane
ground_projection_3d = utils.project_to_xz_plane(body)
utils.visualize(ground_projection_3d)

# Compute the body floor projection contour
points_3d = np.asarray(ground_projection_3d.points)
points_2d = points_3d[:, [0, 2]] # X and Z coordinates
utils.get_body_contour(points_2d)











