import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from scipy.ndimage import center_of_mass

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
frames = pipe.wait_for_frames()
depth_frame = frames.get_depth_frame()
frame_shape = (depth_frame.get_width(), depth_frame.get_height())

# Background mask
background = np.ones(frame_shape[::-1], np.uint8)

# Ids of the heads
heads_id = {}

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
        gray *= background # Remove the background

        # List contaning the images of the detected heads and their features in the current frame
        head_features = []

        # List containing the images of the detected heads in the current frame
        head_images = []

        # Set match_found to False for all heads
        for id, value in heads_id.items():
            height, center, area, _ = value
            heads_id[id] = (height, center, area, False)

        # Find the contours
        _,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        nb_heads = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 2000:
                x,y,w,h = cv2.boundingRect(contour)
                border = 10
                # If the head is too close to the border, ignore it
                if x < border or y < border or x+w > frame_shape[0] - border or y+h > frame_shape[1] - border:
                    continue
                
                # Find the max height in the bounding box
                max_height = 0
                min_distance = None
                for i in range(x, x+w):
                    for j in range(y, y+h):
                        distance = depth_frame.get_distance(i, j)
                        height = 2.67 - distance

                        if 1.0 < height < 2.5 and height > max_height:
                            max_height = height
                            min_distance = distance

                # If no height is found, ignore it
                if max_height == 0:
                    continue

                # Post process the depth frame to keep only the head (between min_distance and min_distance + 0.1)
                depth_frame_head  = post_process_height(depth_frame, min_distance=min_distance, max_distance=min_distance + 0.15)
                depth_color_frame_head = colorizer.colorize(depth_frame_head)
                depth_color_image_head = np.asanyarray(depth_color_frame_head.get_data())

                # Apply the bounding box mask
                mask = np.zeros_like(depth_color_image_head)
                cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
                depth_color_image_head = cv2.bitwise_and(depth_color_image_head, mask)
                
                # Binarize the image
                head_image = cv2.inRange(depth_color_image_head, (1, 1, 1), (255, 255, 255))

                # Smooth the edges by applying median filter
                head_image = cv2.medianBlur(head_image, 15)
  
                # Find center of mass
                mass_y, mass_x = np.where(head_image > 0)
                if mass_x.size == 0 or mass_y.size == 0:
                    continue
                cent_x = np.average(mass_x)
                cent_y = np.average(mass_y)
                center = (int(cent_x), int(cent_y))

                # Recalculate the area
                contours, _ = cv2.findContours(head_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0]
                area = cv2.contourArea(contour)

                head_features.append((max_height, center, area))
                head_images.append(head_image)
                nb_heads += 1

                

        # Match the heads found with the ids
        nb_matches = 0

        for height, center, area in head_features:
            match_percentages = {}

            for id, value in heads_id.items():
                m_height, m_center, m_area, m_match_found = value
                if m_match_found:
                    continue
                
                pos_diff = np.linalg.norm(np.array(m_center) - np.array(center))
                area_diff = abs(m_area - area)
                height_diff = abs(m_height - max_height)

                # Set the maximum difference for each feature
                max_pos_diff = 100
                max_area_diff = 2000
                max_height_diff = 0.15

                if pos_diff > max_pos_diff or area_diff > max_area_diff or height_diff > max_height_diff:
                    continue

                # Normalize
                match_percentages[id] = pos_diff

            print(f'Match percentages: {match_percentages}')

            # Find max match
            if len(match_percentages) > 0:
                # Find max value and the key in the dictionary
                id = min(match_percentages, key=match_percentages.get)
                heads_id[id] = (max_height, center, area, True)
                nb_matches += 1

            else:
                # Else, create a new id
                new_id = len(heads_id)
                heads_id[new_id] = (height, center, area, True)

        if nb_heads != nb_matches:
            print(f'Number of heads detected: {nb_heads}, Number of matches: {nb_matches}')

        for id, value in heads_id.items():
            print('------------------------')
            print(f'ID: {id}, Height: {value[0]:.2f} m, Center: {value[1]}, Area: {value[2]}')
            print('------------------------')

        # Draw the height and the center of the head
        for id, value in heads_id.items():
            height, center, area, _ = value
            cv2.circle(depth_color_image, center, 5, (0, 255, 0), -1)
            cv2.putText(depth_color_image, f'ID: {id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Merge all head images
        heads_image = np.zeros((frame_shape[1], frame_shape[0]), np.uint8)
        for img in head_images:
            heads_image = cv2.add(heads_image, img)

        cv2.imshow('Motion detected', gray)
        cv2.imshow('Heads detected', heads_image)
        cv2.imshow('Tracking', depth_color_image)
        key = cv2.waitKey(wait_key)

        if key == ord('q'):
            break

        if key == ord('d'):
            wait_key = 1 if wait_key == 0 else 0

finally:
    pipe.stop()
    cv2.destroyAllWindows()