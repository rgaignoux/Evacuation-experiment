import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import cv2
from concave_hull import concave_hull, concave_hull_indexes
import utils

# Camera settings
width = 640
height = 480
fps = 15

def crop_body(pcd, body_center):
    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    center_bbox_min = body_center - np.array([0.5, 0.5, 0])
    center_bbox_max = body_center + np.array([0.5, 0.5, 0])

    # Compute z_min as the first percentile in case the actual min_bound is an outlier
    z_values = np.asarray(pcd.points)[:, 2]
    z_min = np.percentile(z_values, 1)

    center_bbox_min[2] = z_min + 0.7 # 0.4 to remove the floor
    center_bbox_max[2] = entire_bbox.max_bound[2]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    # Remove outliers
    cl, ind = body.remove_statistical_outlier(nb_neighbors=100, std_ratio=3)
    body = body.select_by_index(ind)

    return body

capture = False

if capture:
    pipe = utils.create_pipeline("815412070846")

    # Wait 5 seconds
    for i in range(5):
        print("Capture will start in ", 5-i, " seconds")
        time.sleep(1)

    # Save the PLY files
    utils.save_ply("top.ply", pipe)

# Load front point cloud
top_path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\1_camera\\backpack_in_hands\\top.ply"
pcd = o3d.io.read_point_cloud(top_path)
pcd = pcd.voxel_down_sample(voxel_size=0.005)

# Remove outliers and visualize
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)
utils.visualize(pcd)

# Crop the body
body_center = np.array([0, 0.1, -0.7])
body = crop_body(pcd, body_center)
utils.visualize(body)
o3d.io.write_point_cloud("body.ply", body)

# Project the body to the XY plane
ground_projection_3d = utils.project_to_xy_plane(body)
utils.visualize(ground_projection_3d)
    
# Compute the body floor projection contour
points_3d = np.asarray(ground_projection_3d.points)
points_2d = points_3d[:, [0, 1]] # X and Y coordinates
utils.get_body_contour(points_2d)












