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

def crop_body(pcd, center, angle):
    rad = angle * np.pi / 180
    R = pcd.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    center_bbox_min = center - np.array([0.4, 0, 0.4])
    center_bbox_max = center + np.array([0.4, 0, 0.4])

    # Compute y_min as the first percentile in case the actual min_bound is an outlier
    y_values = np.asarray(pcd.points)[:, 1]
    y_min = np.percentile(y_values, 1)

    center_bbox_min[1] = y_min + 0.2 # 0.18 to remove the floor
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    return body


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

# front_pipe = create_pipeline("815412070846")
# side_pipe = create_pipeline("815412070753")

# # Wait 5 seconds
# for i in range(5):
#     print("Capture will start in ", 5-i, " seconds")
#     time.sleep(1)

# # Save the PLY files
# save_ply("front.ply", front_pipe)
# save_ply("side.ply", side_pipe)

# Load front point cloud
pcd_front = o3d.io.read_point_cloud("front.ply")
pcd_front = pcd_front.uniform_down_sample(every_k_points=5)

# Load side point cloud
pcd_side = o3d.io.read_point_cloud("side.ply")
pcd_side = pcd_side.uniform_down_sample(every_k_points=5)

visualize(pcd_front)
visualize(pcd_side)

# Crop the body
front_body_center = np.array([0, 0, -1.5])
front_angle = -16
front_body = crop_body(pcd_front, front_body_center, front_angle)

side_body_center = np.array([0, 0, -1.2])
side_angle = -19
side_body = crop_body(pcd_side, side_body_center, side_angle)

visualize(front_body)
visualize(side_body)
