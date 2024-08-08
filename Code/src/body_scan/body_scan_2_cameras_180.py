import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import utils

# Read translation & angles values from translation_rotation_180.txt
filename = 'translation_rotation_180.txt'
with open(filename) as file:
    lines = [line.rstrip() for line in file]

translation = lines[0].split()
front_angle = float(lines[1])
back_angle = float(lines[2])

def crop_body(pcd_front, pcd_back, body_center):
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

    center_bbox_min[1] = y_min + 0.2 # 0.2 to remove the floor
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    # Remove outliers
    cl, ind = body.remove_statistical_outlier(nb_neighbors=64, std_ratio=1.8)
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

# Load front point cloud
pcd_front = o3d.io.read_point_cloud("front.ply")
pcd_front = pcd_front.uniform_down_sample(every_k_points=5)

# Load back point cloud
pcd_back = o3d.io.read_point_cloud("back.ply")
pcd_back = pcd_back.uniform_down_sample(every_k_points=5)

# Remove outliers and visualize
cl, ind = pcd_front.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_front = pcd_front.select_by_index(ind)
utils.visualize(pcd_front)

cl, ind = pcd_back.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_back = pcd_back.select_by_index(ind)
utils.visualize(pcd_back)

# Get the body only
body_center = np.array([0, 0, 1.2]) # Center of the body : computed manually by pressing shift + left click in the visualization window
body = crop_body(pcd_front, pcd_back, body_center)
utils.visualize(body)
o3d.io.write_point_cloud("body.ply", body)

# Project the body to the XZ plane
ground_projection = utils.project_to_xz_plane(body)
utils.visualize(ground_projection)

# Get bounding box
body_bbox = ground_projection.get_axis_aligned_bounding_box()
body_dimensions = body_bbox.get_max_bound() - body_bbox.get_min_bound()
print("Body dimensions: ", body_dimensions)

# Save the projection to a file
points = np.asarray(ground_projection.points)
x = points[:, 0]
z = points[:, 2]
plt.figure(figsize=(10, 10))
plt.scatter(x, z, s=0.5, c='b', marker='o')
plt.title('2D Projection on XZ Plane')
plt.xlabel('X')
plt.ylabel('Z')
plt.grid(True)
plt.axis('equal')
plt.savefig("projection.png", dpi=300)
plt.close()










