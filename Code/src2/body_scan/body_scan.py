import pyrealsense2 as rs
import numpy as np
import time
import open3d as o3d
import numpy as np
from post_process import post_process_depth_frame

# Camera settings
width = 640
height = 480
fps = 30

# Translation / Rotation values
distance_cameras = -2.6
angles_cameras = -22

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
    center_bbox_min[1] = entire_bbox.min_bound[1]
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

    return pipeline

def save_ply(path, pipe):
    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        # Create save_to_ply object
        ply = rs.save_to_ply(path)

        print("Saving PLY to ", path)
        # Apply the processing block to the frameset which contains the depth frame and the texture
        ply.process(colorized)
        print("Done")
    finally:
        pipe.stop()

front_pipe = create_pipeline("815412070753")
back_pipe = create_pipeline("815412070846")

# Wait for auto-exposure to stabilize
for x in range(50):
    front_pipe.wait_for_frames()
    back_pipe.wait_for_frames()

# Wait 5 seconds
for i in range(5):
    print("Capture will start in ", 5-i, " seconds")
    time.sleep(1)

# Save the PLY files
save_ply("front.ply", front_pipe)
save_ply("back.ply", back_pipe)

# Load front point cloud
pcd_front = o3d.io.read_point_cloud("front.ply")
pcd_front = pcd_front.uniform_down_sample(every_k_points=5)

# Load back point cloud
pcd_back = o3d.io.read_point_cloud("back.ply")
pcd_back = pcd_back.uniform_down_sample(every_k_points=5)

# Remove outliers and visualize
cl, ind = pcd_front.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_front = pcd_front.select_by_index(ind)
visualize(pcd_front)

cl, ind = pcd_back.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_back = pcd_back.select_by_index(ind)
visualize(pcd_back)

body = crop_body(pcd_front, pcd_back, distance_cameras, angles_cameras)
visualize(body)









