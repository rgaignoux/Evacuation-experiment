import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from post_process import post_process_depth_frame


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
bag_file = bag_file.replace('\\', '\\\\')

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False) # False: no frame drop
colorizer = rs.colorizer(2)
wait_key = 30
frame_shape = (848, 480)

background = cv2.imread('background.png', cv2.IMREAD_GRAYSCALE)

ids_heights_positions = {}

try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame_pp = post_process_depth_frame(depth_frame, 0, 2)
        depth_color_frame = colorizer.colorize(depth_frame_pp)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        
        # Motion detection
        gray = cv2.cvtColor(depth_color_image,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (35, 35), 0)
        frameDelta = cv2.absdiff(background, gray)

        # Find the contours
        _,thresh = cv2.threshold(frameDelta,100,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            print("Area: ", area)

            if area > 2000:
                x,y,w,h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)

                # Find the max height in the bounding box
                max_height = 0
                min_distance = 0
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

                depth_frame_pp2  = post_process_height(depth_frame, )
                depth_color_frame2 = colorizer.colorize(depth_frame_pp2)
                depth_color_image2 = np.asanyarray(depth_color_frame2.get_data())
                # Apply mask : bounding box
                mask = np.zeros_like(depth_color_image2)
                cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
                depth_color_image2 = cv2.bitwise_and(depth_color_image2, mask)

                cv2.imshow('depthpp',depth_color_image2)

                # Look if there is a match in the existing IDs
                matching_id = -1
                match_proba = 0
                for key, value in ids_heights_positions.items():
                    (height, pos) = value
                    delta_height = abs(height - max_height)
                    delta_pos = np.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)

                    max_delta_height = 0.3
                    max_delta_pos = 70

                    if delta_pos > max_delta_pos or delta_height > max_delta_height:
                        continue

                    # Normalize
                    delta_height /= max_delta_height
                    delta_pos /= max_delta_pos

                    proba = 1 - (0.5 * delta_height + 0.5 * delta_pos)

                    if proba > match_proba:
                        matching_id = key
                        match_proba = proba

                

                if match_proba < 0.5 or matching_id == -1:
                    # Create a new ID
                    ids_heights_positions[len(ids_heights_positions)] = (max_height, center)
                # If pos is close to the border, ignore it
                elif match_proba > 0.5 and x < 50 or x+w > frame_shape[0] - 50 or y < 50 or y+h > frame_shape[1] - 50:
                    del ids_heights_positions[matching_id]

                else:
                    # Update the ID
                    ids_heights_positions[matching_id] = (max_height, center)

                cv2.rectangle(depth_color_image,(x,y),(x+w,y+h),(255,0,255),3)
                
                for key, value in ids_heights_positions.items():
                    (height, pos) = value
                    cv2.circle(depth_color_image, pos, 5, (0, 0, 255), -1)
                    cv2.putText(depth_color_image, str(round(height, 2)), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print("IDs: ", ids_heights_positions)
        cv2.imshow('VIDEO',depth_color_image)
        cv2.imshow('threshold',thresh)
        key = cv2.waitKey(wait_key)

        if key == ord('q'):
            break

        if key == ord('d'):
            wait_key = 30 if wait_key == 0 else 0

finally:
    pipe.stop()
    cv2.destroyAllWindows()