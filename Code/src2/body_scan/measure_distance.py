import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def post_process_depth_frame(depth_frame, min_distance=0, max_distance=4.0, decimation_magnitude = 1.0, spatial_magnitude = 2.0, spatial_smooth_alpha = 0.5, spatial_smooth_delta = 20, temporal_smooth_alpha = 0.4, temporal_smooth_delta = 20):
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
    filtered_frame = hole_filling.process(filtered_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
    return file_path

def select_point(event, x, y, flags, param):
    image, points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', image)

def pick_points(image):
    points = []
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', select_point, [image, points])
    while True:
        cv2.imshow('image', image)
        if len(points) == 2:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return points

# Read the bag file
bag_file = select_file()

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False) # False: no frame drop

wait_key = 1
pick = False

try:
    while True:
        frames = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = post_process_depth_frame(depth_frame)
        colorizer = rs.colorizer()
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.imshow('Depth Image', depth_color_image)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Color Image', color_image)

        if(pick):
            # Pick two points
            points = pick_points(color_image)

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            u = points[0]
            v = points[1]

            udist = depth_frame.get_distance(u[0], u[1])
            vdist = depth_frame.get_distance(v[0], v[1])
            print("Distance at u: ", udist)
            max_dist = 2.4

            point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [u[0], u[1]], max_dist)
            point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [v[0], v[1]], max_dist)

            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            print("Distance between points: ", distance)

            pick = False

        key = cv2.waitKey(wait_key)

        if key == ord('p'):
            pick = not pick

        if key == ord('q'):
            break

        if key == ord('s'):
            cv2.imwrite('image.png', color_image)

        if key == ord('d'):
            wait_key = 0 if wait_key == 1 else 1

finally:
    pipe.stop()
    cv2.destroyAllWindows()