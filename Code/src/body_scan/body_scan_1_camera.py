import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import utils

# Camera settings
width = 640
height = 480
fps = 15

def crop_body(pcd, body_center):
    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    center_bbox_min = body_center - np.array([0.4, 0.4, 0])
    center_bbox_max = body_center + np.array([0.4, 0.4, 0])

    # Compute y_min as the first percentile in case the actual min_bound is an outlier
    z_values = np.asarray(pcd.points)[:, 2]
    z_min = np.percentile(z_values, 1)

    center_bbox_min[2] = z_min + 0.2 # 0.18 to remove the floor
    center_bbox_max[2] = entire_bbox.max_bound[2]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    return body

pipe = utils.create_pipeline("815412070846")

# Wait 5 seconds
for i in range(5):
    print("Capture will start in ", 5-i, " seconds")
    time.sleep(1)

# Save the PLY files
utils.save_ply("top.ply", pipe)

# Load front point cloud
pcd = o3d.io.read_point_cloud("top.ply")
pcd = pcd.uniform_down_sample(every_k_points=3)

# Remove outliers and visualize
cl, ind = pcd.remove_radius_outlier(nb_points=35, radius=0.05)
pcd = pcd.select_by_index(ind)
utils.visualize(pcd)

# Crop the body
body_center = np.array([0, 0.1, -0.7])
body = crop_body(pcd, body_center)
utils.visualize(body)

# Project the body to the XZ plane
ground_projection = utils.project_to_xy_plane(body)
utils.visualize(ground_projection)

# Get bounding box
body_bbox = ground_projection.get_axis_aligned_bounding_box()
body_dimensions = body_bbox.get_max_bound() - body_bbox.get_min_bound()
print("Body dimensions: ", body_dimensions)
    
# Save the plot to a file
points = np.asarray(ground_projection.points)
x = points[:, 0]
y = points[:, 1]
plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=0.5, c='b', marker='o')
plt.title('2D Projection on XZ Plane')
plt.xlabel('X')
plt.ylabel('Z')
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes
plt.savefig("projection.png", dpi=300)
plt.close()










