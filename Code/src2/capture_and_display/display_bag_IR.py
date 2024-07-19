import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from post_process import post_process_depth_frame

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
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
align = rs.align(rs.stream.depth)
colorizer = rs.colorizer(2)
wait_key = 30

try:
    while True:
        frames = pipe.wait_for_frames()

        infrared_frame1 = frames.get_infrared_frame()

        aligned_frames = align.process(frames)

        depth_frame_aligned = aligned_frames.get_depth_frame()
        infrared_frame2 = aligned_frames.get_infrared_frame() # left
        
        #depth_color_frame = colorizer.colorize(depth_frame_aligned)
        #depth_color_image = np.asanyarray(depth_color_frame.get_data())

        infrared_image1 = np.asanyarray(infrared_frame1.get_data())
        infrared_image2 = np.asanyarray(infrared_frame2.get_data())

        # resize by 1.5
        infrared_image1 = cv2.resize(infrared_image1, (0, 0), fx=0.75, fy=0.75)
        infrared_image2 = cv2.resize(infrared_image2, (0, 0), fx=0.75, fy=0.75)

        images = np.hstack((infrared_image1, infrared_image2))
        cv2.imshow('IR', images)

        key = cv2.waitKey(wait_key)
        if key == ord('q'):
            break

        if key == ord('s'):
            cv2.imwrite('background.png', depth_color_image)

        if key == ord('d'):
            wait_key = 30 if wait_key == 0 else 0

        

finally:
    pipe.stop()
    cv2.destroyAllWindows()