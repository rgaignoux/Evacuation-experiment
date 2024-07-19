import cv2
import pyrealsense2 as rs
import numpy as np
import tkinter as tk
from tkinter import filedialog
from post_process import post_process_depth_frame

def read_bag_file(file_name, real_time=False):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(file_name, repeat_playback=False)
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(real_time) # False: no frame drop

    # Frame shape
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    frame_shape = (depth_frame.get_height(), depth_frame.get_width())

    return pipe, cfg, profile, playback, frame_shape

def post_process_depth_frame(depth_frame, min_distance, max_distance):
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

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la fenêtre de dialogue pour sélectionner un fichier
    return file_path

# Read the bag file
bag_file = select_file()
bag_file = bag_file.replace('\\', '\\\\')
pipe, cfg, profile, playback, frame_shape = read_bag_file(bag_file)
wait_key = 1

frames = pipe.wait_for_frames()
depth_frame = frames.get_depth_frame()
colorizer = rs.colorizer()
depth_color_frame = colorizer.colorize(depth_frame)
depth_color_image = np.asanyarray(depth_color_frame.get_data())

# Min and max distance
actual_min_distance = 0
actual_max_distance = 4

# ROI
refPt = [(0, 0), (frame_shape[1], frame_shape[0])]
cropping = False
mask = np.ones(frame_shape, dtype=np.uint8)

def show_depth_image(min_distance, max_distance):
    global depth_frame
    global depth_color_image
    global mask

    new_depth_frame = post_process_depth_frame(depth_frame, min_distance, max_distance)

    # Colorize the depth frame
    depth_color_frame = colorizer.colorize(new_depth_frame)

    # Convert frames to images
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # Apply the mask
    #depth_color_image = cv2.bitwise_and(depth_color_image, depth_color_image, mask=mask)

    cv2.imshow('Depth Image', depth_color_image)

def on_trackbar_min(val):
    global actual_min_distance
    global depth_frame
    cm = val / 100
    actual_min_distance = cm

    print("Max : {} | Min : {}".format(actual_max_distance, actual_min_distance))
    
    show_depth_image(actual_min_distance, actual_max_distance)

def on_trackbar_max(val):
    global actual_max_distance
    global depth_frame
    cm = val / 100
    actual_max_distance = cm

    print("Max : {} | Min : {}".format(actual_max_distance, actual_min_distance))

    show_depth_image(actual_min_distance, actual_max_distance)

def find_max(val):
    global depth_frame, refPt
    if val == 1:
        min = (0, 0, 10000000)
        for i in range(depth_frame.get_width()):
            for j in range(depth_frame.get_height()):

                if i < refPt[0][0] or i > refPt[1][0] or j < refPt[0][1] or j > refPt[1][1]:
                    continue
                
                distance = depth_frame.get_distance(i, j)
                if distance != 0.0 and distance < min[2]:
                    min = (i, j, distance)

        print("Min : {} | Max : {}".format(min[2], min[2] + 0.1))

        show_depth_image(min[2], min[2] + 0.1)

def set_ROI(event, x, y, flags, param):
    global refPt, cropping, mask

    # Si le bouton gauche de la souris est cliqué, enregistrer le point de départ (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # Si le bouton gauche de la souris est relâché, enregistrer le point de fin (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

    if not cropping:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        print(refPt)
        cv2.rectangle(mask, refPt[0], refPt[1], 255, -1)
        show_depth_image(actual_min_distance, actual_max_distance)
    

cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Min', 'Depth Image', int(actual_min_distance * 100), 400, on_trackbar_min)
cv2.createTrackbar('Max', 'Depth Image', int(actual_max_distance * 100), 400, on_trackbar_max)
cv2.createTrackbar('Find max', 'Depth Image', 0, 1, find_max)
cv2.setMouseCallback("Depth Image", set_ROI)

try:
    while True:
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()

        depth_frame = frames.get_depth_frame()

        # Post process is not included in the BAG file, so we need to apply it
        depth_frame = post_process_depth_frame(depth_frame, actual_min_distance, actual_max_distance)

        # Colorize the depth frame
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert frames to images
        depth_color_image = np.asanyarray(depth_frame.get_data())

        # Apply the mask
        #depth_color_image = cv2.bitwise_and(depth_color_image, depth_color_image, mask=mask)

        cv2.imshow('Depth Image', depth_color_image)

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


