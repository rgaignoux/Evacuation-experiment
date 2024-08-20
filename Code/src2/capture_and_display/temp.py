import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

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

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
    return file_path

def read_bag_file(file_name, real_time=False):
    """
    Read a bag file recorded from a RealSense camera.

    Parameters:
        file_name (str): the path to the bag file.
        real_time (bool): if True, the playback is in real time (frames may be dropped).
    """
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(file_name, repeat_playback=False)
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(real_time) # False: no frame drop
    
    # Get the frame shape
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    frame_shape = (depth_frame.get_height(), depth_frame.get_width())

    return pipe, cfg, profile, playback, frame_shape


# Read the bag file
path = select_file()
pipe, cfg, profile, playback, frame_shape = read_bag_file(path)
# Get frame number = 244595 (bg)
depth_array_bg = None

while(True):
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    frame_number = depth_frame.get_frame_number()
    if frame_number == 244595:
        depth_frame = post_process_depth_frame(depth_frame, fill_hole=True)
        depth_array_bg = np.asanyarray(depth_frame.get_data()) / 1000
        break

pipe.stop()

walls = np.ones(frame_shape, dtype=np.uint8)
walls[320:, 0:300] = 0
walls[320:, 540:] = 0
depth_array_bg = depth_array_bg * walls

# Read the bag file
pipe, cfg, profile, playback, frame_shape = read_bag_file(path, real_time=False)
wait_key = 1

while(True):
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_frame = post_process_depth_frame(depth_frame, fill_hole=True)
    depth_array = np.asanyarray(depth_frame.get_data()) / 1000

    depth_color_frame = rs.colorizer().colorize(depth_frame)
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # Pixel (i,j) is set to foreground if the difference between the depth value at (i,j) and the background depth value at (i,j) is greater than 0.5
    diff = abs(depth_array_bg - depth_array)
    diff[diff < 1] = 0

    cv2.imshow('Foreground', diff)
    cv2.imshow('Depth Image', depth_color_image)

    key = cv2.waitKey(wait_key)
    if key == ord('q'):
        break
    if key == ord('d'):
        wait_key = 0 if wait_key == 1 else 1
    if key == ord('s'):
        cv2.imwrite('image.png', depth_color_image)
