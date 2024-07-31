import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
from post_process import post_process_depth_frame

# Camera settings
width = 640
height = 480
fps = 15

def visualize(pcd):
    viewer = o3d.visualization.VisualizerWithEditing()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    print(viewer.get_picked_points())

def crop_body(pcd):
    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    body_center = np.array([0, 0.1, -0.7])
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

def project_to_xy_plane(pcd):
    # Get points from the point cloud
    points = np.asarray(pcd.points)
    
    # Set Z coordinate to 0
    points[:, 2] = 0
    
    # Update point cloud with new points
    pcd.points = o3d.utility.Vector3dVector(points)

    # Remove outliers
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
    pcd = pcd.select_by_index(ind)
    
    return pcd

def create_pipeline():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Set emitter enabled to "laser"
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Use Medium Density preset
    depth_sensor.set_option(rs.option.visual_preset, 5)

    # Set manual exposure to 5000
    depth_sensor.set_option(rs.option.exposure, 5000)

    return pipeline

def save_ply(path, pipe):
    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = post_process_depth_frame(depth_frame, min_distance=0, max_distance=3, temporal_smooth_alpha=0.2, temporal_smooth_delta=40)

        # Create save_to_ply object
        ply = rs.save_to_ply(path)
        print("Saving PLY to ", path)
        ply.process(depth_frame)
        print("Done")
    finally:
        pipe.stop() 

pipe = create_pipeline()

# Wait 5 seconds
for i in range(5):
    print("Capture will start in ", 5-i, " seconds")
    time.sleep(1)

# Save the PLY files
save_ply("top.ply", pipe)

# Load front point cloud
pcd = o3d.io.read_point_cloud("top.ply")
pcd = pcd.uniform_down_sample(every_k_points=3)

# Remove outliers and visualize
cl, ind = pcd.remove_radius_outlier(nb_points=35, radius=0.05)
pcd = pcd.select_by_index(ind)
visualize(pcd)

# Crop the body
body = crop_body(pcd)
visualize(body)

# Project the body to the XZ plane
ground_projection = project_to_xy_plane(body)
visualize(ground_projection)

# Get bounding box
body_bbox = ground_projection.get_axis_aligned_bounding_box()
body_dimensions = body_bbox.get_max_bound() - body_bbox.get_min_bound()
print("Body dimensions: ", body_dimensions)

points = np.asarray(ground_projection.points)
    
# Extract X and Y coordinates
x = points[:, 0]
y = points[:, 1]

# Create the plot
plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=0.5, c='b', marker='o')
plt.title('2D Projection on XZ Plane')
plt.xlabel('X')
plt.ylabel('Z')
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes

# Save the plot to a file
plt.savefig("projection.png", dpi=300)
plt.close()










