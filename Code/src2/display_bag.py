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

try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = post_process_depth_frame(depth_frame, 0, 2)
        colorizer = rs.colorizer(2)
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.imshow('Depth Image', depth_color_image)

        wait_key = cv2.waitKey(1)
        if wait_key == ord('q'):
            break

        if wait_key == ord('s'):
            cv2.imwrite('background.png', depth_color_image)

finally:
    pipe.stop()
    cv2.destroyAllWindows()