import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def select_path():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    return path

def capture_IR_image(serial_number, path):
    #path = select_path()
    print("Selected path:", path)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial_number)
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 6)
    profile = pipe.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.exposure, 7000)

    frames = pipe.wait_for_frames()

    # Pass 30 frames to allow the camera to adjust brightness
    for i in range(30):
        frames = pipe.wait_for_frames()

    infrared_frame = frames.get_infrared_frame()
    infrared_image = np.asanyarray(infrared_frame.get_data())
    cv2.imwrite(path, infrared_image)

    pipe.stop()
    print("Image saved to", path)

serial_number1 = "815412070753"
path = select_path()
capture_IR_image(serial_number1, path)

serial_number2 = "815412070846"
path = select_path()
capture_IR_image(serial_number2, path)