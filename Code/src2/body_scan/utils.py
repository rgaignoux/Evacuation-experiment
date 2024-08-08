import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
    return file_path

# Red : x-axis
# Green : y-axis
# Blue : z-axis
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


# Camera settings
width = 640
height = 480
fps = 15


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
    depth_sensor.set_option(rs.option.exposure, 8500)

    return pipeline


def post_process_depth_frame(depth_frame, min_distance=0, max_distance=4, decimation_magnitude = 1.0, spatial_magnitude = 2.0, spatial_smooth_alpha = 0.5, spatial_smooth_delta = 20, temporal_smooth_alpha = 0.4, temporal_smooth_delta = 20, fill_hole = False):
    """
    Apply post processing filters to the depth frame of a RealSense camera.
    More information about the filters can be found at:
    - https://dev.intelrealsense.com/docs/post-processing-filters
    - https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
    - https://intelrealsense.github.io/librealsense/doxygen/namespacers2.html
    """
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters
    decimation_filter = rs.decimation_filter()
    threshold_filter = rs.threshold_filter()
    depth_to_disparity = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(1) # https://intelrealsense.github.io/librealsense/doxygen/classrs2_1_1hole__filling__filter.html

    # Apply the control parameters for the filters
    decimation_filter.set_option(rs.option.filter_magnitude, decimation_magnitude)
    threshold_filter.set_option(rs.option.min_distance, min_distance)
    threshold_filter.set_option(rs.option.max_distance, max_distance)
    spatial_filter.set_option(rs.option.filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(rs.option.filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(rs.option.filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(rs.option.filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    # Post processing order : https://dev.intelrealsense.com/docs/post-processing-filters
    # Depth Frame >> Decimation Filter >> Depth2Disparity Transform >> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform >> Hole Filling Filter >> Filtered Depth
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = threshold_filter.process(filtered_frame)
    filtered_frame = depth_to_disparity.process(filtered_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)
    filtered_frame = disparity_to_depth.process(filtered_frame)
    if fill_hole:
        filtered_frame = hole_filling.process(filtered_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered


def save_ply(path, pipe):
    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = post_process_depth_frame(depth_frame, temporal_smooth_alpha=0.2, temporal_smooth_delta=40)

        # Create save_to_ply object
        ply = rs.save_to_ply(path)
        print("Saving PLY to ", path)
        ply.process(depth_frame)
        print("Done")
    finally:
        pipe.stop() 