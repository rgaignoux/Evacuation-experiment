import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def post_process_depth_frame(depth_frame, min_distance=0, max_distance=3.0, decimation_magnitude = 1.0, spatial_magnitude = 2.0, spatial_smooth_alpha = 0.5, spatial_smooth_delta = 20, temporal_smooth_alpha = 0.4, temporal_smooth_delta = 20):
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
    #filtered_frame = hole_filling.process(filtered_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered


def post_process_height(depth_frame, min_distance, max_distance):
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    threshold_filter = rs.threshold_filter()

    threshold_filter.set_option(rs.option.min_distance, min_distance)
    threshold_filter.set_option(rs.option.max_distance, max_distance)

    filtered_frame = threshold_filter.process(depth_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered


def select_file():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    return file_path

# Read the bag file
bag_file = select_file()
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False) # False: no frame drop
colorizer = rs.colorizer(2)
wait_key = 1
frame_shape = (848, 480)

background = np.ones((480, 848), np.uint8) * 1
cv2.rectangle(background,(0,0),(210,250), 0,-1)


try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame_pp = post_process_depth_frame(depth_frame, min_distance=0, max_distance=1.2)
        depth_color_frame = colorizer.colorize(depth_frame_pp)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        
        # Motion detection
        gray = cv2.cvtColor(depth_color_image,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (35, 35), 0)
        frameDelta = gray * background

        depth_images_merged = []

        # Find the contours
        _,thresh = cv2.threshold(frameDelta,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            #print("Area: ", area)

            if area > 2000:
                x,y,w,h = cv2.boundingRect(contour)
                
                # Find the max height in the bounding box
                max_height = 0
                min_distance = None
                for i in range(x, x+w):
                    for j in range(y, y+h):
                        distance = depth_frame.get_distance(i, j)
                        height = 2.67 - distance

                        if 1.5 < height < 2.0 and height > max_height:
                            max_height = height
                            min_distance = distance

                # If no height is found, ignore it
                if max_height == 0:
                    continue

                # Post process the depth frame to keep only the head (between min_distance and min_distance + 0.1)
                depth_frame_pp2  = post_process_height(depth_frame, min_distance=min_distance, max_distance=min_distance + 0.1)
                depth_color_frame2 = colorizer.colorize(depth_frame_pp2)
                depth_color_image2 = np.asanyarray(depth_color_frame2.get_data())
                mask = np.zeros_like(depth_color_image2)
                cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
                depth_color_image2 = cv2.bitwise_and(depth_color_image2, mask)
                depth_color_image_gray = cv2.cvtColor(depth_color_image2, cv2.COLOR_BGR2GRAY)
                
                # Find center of mass
                mass_x, mass_y = np.where(depth_color_image_gray > 0)
                cent_x = np.average(mass_x)
                cent_y = np.average(mass_y)
                center = (int(cent_y), int(cent_x))
                cv2.circle(depth_color_image2, center, 5, (0, 0, 255), -1)

                depth_images_merged.append(depth_color_image2)

        # Merge all frames in the list depth_images_pp
        merged_image = np.zeros_like(depth_color_image)
        for img in depth_images_merged:
            merged_image = cv2.add(merged_image, img)

        depth_color_image = cv2.resize(depth_color_image, (0, 0), fx=0.75, fy=0.75)
        merged_image = cv2.resize(merged_image, (0, 0), fx=0.75, fy=0.75)
        images = np.hstack((depth_color_image, merged_image))
        cv2.imshow('images',images)
        key = cv2.waitKey(wait_key)

        if key == ord('q'):
            break

        if key == ord('d'):
            wait_key = 1 if wait_key == 0 else 0

finally:
    pipe.stop()
    cv2.destroyAllWindows()