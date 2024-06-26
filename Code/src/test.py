import cv2
import pyrealsense2 as rs
import numpy as np

def read_bag_file(file_name, real_time=False):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(file_name, repeat_playback=False)
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(real_time) # False: no frame drop
    
    # Get the frame shape of the color sensor
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame_shape = (color_frame.get_height(), color_frame.get_width())

    return pipe, cfg, profile, playback, frame_shape

def post_process_depth_frame(depth_frame, min_distance=0, max_distance=1.5):
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    decimation_magnitude = 1.0
    spatial_magnitude = 2.0
    spatial_smooth_alpha = 0.5
    spatial_smooth_delta = 20
    temporal_smooth_alpha = 0.4
    temporal_smooth_delta = 20

    # Available filters
    decimation_filter = rs.decimation_filter()
    threshold_filter = rs.threshold_filter()
    depth_to_disparity = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)

    # Options for hole_filling_filter
    # 0 - fill_from_left - Use the value from the left neighbor pixel to fill the hole
    # 1 - farest_from_around - Use the value from the neighboring pixel which is furthest away from the sensor
    # 2 - nearest_from_around - Use the value from the neighboring pixel closest to the sensor
    hole_filling = rs.hole_filling_filter(1)

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
    # Depth Frame >> Decimation Filter >> Depth2Disparity Transform >> Spatial Filter 
    # >> Temporal Filter >> Disparity2Depth Transform >> Hole Filling Filter >> Filtered Depth
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = threshold_filter.process(filtered_frame)
    filtered_frame = depth_to_disparity.process(filtered_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)
    filtered_frame = disparity_to_depth.process(filtered_frame)
    #filtered_frame = hole_filling.process(filtered_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered


# Read the bag file
bag_file = 'C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\me_and_sasa.bag'

pipe, cfg, profile, playback, frame_shape = read_bag_file(bag_file)

colorizer = rs.colorizer()

align = rs.align(rs.stream.color)
wait_key = 1

depth_frame = None
depth_color_frame = None
depth_color_image = None
aligned_depth_frame = None
actual_min_distance = 0
actual_max_distance = 1.5

def on_trackbar_min(val):
    cm = val / 100
    global actual_min_distance
    actual_min_distance = cm
    depth_frame = post_process_depth_frame(aligned_depth_frame, cm, actual_max_distance)

    # Colorize the depth frame
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert frames to images
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    cv2.imshow('Depth Image', depth_color_image)

def on_trackbar_max(val):
    cm = val / 100
    global actual_max_distance
    actual_max_distance = cm
    depth_frame = post_process_depth_frame(aligned_depth_frame, actual_min_distance, cm)

    # Colorize the depth frame
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert frames to images
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    cv2.imshow('Depth Image', depth_color_image)

cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Min', 'Depth Image', int(actual_min_distance * 100), 400, on_trackbar_min)
cv2.createTrackbar('Max', 'Depth Image', int(actual_max_distance * 100), 400, on_trackbar_max)

try:
    while True:
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Post process is not included in the BAG file, so we need to apply it
        depth_frame = post_process_depth_frame(aligned_depth_frame, actual_min_distance, actual_max_distance)

        # Colorize the depth frame
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert frames to images
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('Depth Image', depth_color_image)
        cv2.imshow('Color Image', color_image)

        key = cv2.waitKey(wait_key)

        # Press esc close the image window
        if key == 27:
            break

        # Press d to view the video frame by frame
        if key == ord('d'):
            if wait_key == 0:
                wait_key = 1
            else:
                wait_key = 0


# Catch exception if the stream is ended
except RuntimeError:
    print("Stream ended")
        
finally:
    # Stop streaming
    cv2.destroyAllWindows()
    pipe.stop()


img = cv2.imread('C:\\Users\\Robin\\Pictures\\test.png')
ellipses = cv2.ximgproc.findEllipses(img)