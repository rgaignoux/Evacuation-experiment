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

# Translation / Rotation values
distance_cameras = -2.56
angles_cameras = -25

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

def crop_body(pcd_front, pcd_back, distance_cameras, angle_cameras):
    # Rotate front pcd by 180 degrees on Y axis
    R = pcd_front.get_rotation_matrix_from_xyz((0, np.pi, 0))
    pcd_front.rotate(R, center=(0, 0, 0))

    # Rotate both pcds by camera_angles degrees on X axis
    rad = angle_cameras * np.pi / 180
    R = pcd_front.get_rotation_matrix_from_xyz((-rad, 0, 0))
    pcd_front.rotate(R, center=(0, 0, 0))

    R = pcd_back.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd_back.rotate(R, center=(0, 0, 0))

    # Translate front pcd by distance_between_cameras on Z axis
    pcd_front.translate((0, 0, distance_cameras))

    # Merge both meshes
    pcd = pcd_front + pcd_back

    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    body_center = np.array([0, 0, -1.2])
    center_bbox_min = body_center - np.array([0.5, 0, 0.5])
    center_bbox_max = body_center + np.array([0.5, 0, 0.5])

    # Compute y_min as the first percentile in case the actual min_bound is an outlier
    y_values = np.asarray(pcd.points)[:, 1]
    y_min = np.percentile(y_values, 1)

    center_bbox_min[1] = y_min + 0.18 # 0.18 to remove the floor
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    return body

def project_to_xz_plane(pcd):
    # Get points from the point cloud
    points = np.asarray(pcd.points)
    
    # Set Y coordinate to 0
    points[:, 1] = 0
    
    # Update point cloud with new points
    pcd.points = o3d.utility.Vector3dVector(points)

    # Remove outliers
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
    pcd = pcd.select_by_index(ind)
    
    return pcd

def create_pipeline(serial_number):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_device(serial_number)
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

# back_pipe = create_pipeline("815412070753")
# front_pipe = create_pipeline("815412070846")

# # Wait 5 seconds
# for i in range(5):
#     print("Capture will start in ", 5-i, " seconds")
#     time.sleep(1)

# # Save the PLY files
# save_ply("front.ply", front_pipe)
# save_ply("back.ply", back_pipe)

# Load front point cloud
pcd_front = o3d.io.read_point_cloud("C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras\\walking\\not_facing_cameras\\front.ply")
pcd_front = pcd_front.uniform_down_sample(every_k_points=5)

# Load back point cloud

#pcd_back = o3d.io.read_point_cloud("back.ply")
pcd_back = o3d.io.read_point_cloud("C:\\Users\\Robin\\Documents\\Stage2024\\Code\\body_scans\\2_cameras\\walking\\not_facing_cameras\\back.ply")
pcd_back = pcd_back.uniform_down_sample(every_k_points=5)

# Remove outliers and visualize
cl, ind = pcd_front.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_front = pcd_front.select_by_index(ind)
visualize(pcd_front)

cl, ind = pcd_back.remove_radius_outlier(nb_points=8, radius=0.05)
pcd_back = pcd_back.select_by_index(ind)
visualize(pcd_back)

body = crop_body(pcd_front, pcd_back, distance_cameras, angles_cameras)
visualize(body)
o3d.io.write_point_cloud("body.ply", body)

# Project the body to the XZ plane
ground_projection = project_to_xz_plane(body)
visualize(ground_projection)

# Get bounding box
body_bbox = ground_projection.get_axis_aligned_bounding_box()
body_dimensions = body_bbox.get_max_bound() - body_bbox.get_min_bound()
print("Body dimensions: ", body_dimensions)

points = np.asarray(ground_projection.points)
    
# Extract X and Z coordinates
x = points[:, 0]
z = points[:, 2]

# Create the plot
plt.figure(figsize=(10, 10))
plt.scatter(x, z, s=0.5, c='b', marker='o')
plt.title('2D Projection on XZ Plane')
plt.xlabel('X')
plt.ylabel('Z')
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes

# Save the plot to a file
plt.savefig("projection.png", dpi=300)
plt.close()










