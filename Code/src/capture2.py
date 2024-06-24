import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pyrealsense2 as rs
import json

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recorder App")

        # Initialize recording parameters
        self.width = 848
        self.height = 480
        self.color_fps = 60
        self.depth_fps = 60
        self.auto_exposure = True
        
        # Record button
        self.record_button = tk.Button(root, text="Record", command=self.start_recording)
        self.record_button.pack(pady=10)
        
        # Stop button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_recording)
        self.stop_button.pack(pady=10)
        
        # Path selector button for first stream
        self.path_button = tk.Button(root, text="Select Path 1", command=self.select_path1)
        self.path_button.pack(pady=10)
        
        # Label to display the selected path for first stream
        self.path_label1 = tk.Label(root, text="No path selected for Stream 1")
        self.path_label1.pack(pady=10)
        
        # Path selector button for second stream
        self.path_button2 = tk.Button(root, text="Select Path 2", command=self.select_path2)
        self.path_button2.pack(pady=10)
        
        # Label to display the selected path for second stream
        self.path_label2 = tk.Label(root, text="No path selected for Stream 2")
        self.path_label2.pack(pady=10)
        
        self.recording = False
        self.file_path1 = ""
        self.file_path2 = ""
        self.pipeline1 = None
        self.pipeline2 = None


    def start_recording(self):
        if not self.file_path1 or not self.file_path2:
            messagebox.showerror("Error", "Please select paths for both streams.")
            return
        
        if not self.recording:
            self.recording = True
            self.record_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            print(f"Recording started, saving to {self.file_path1} and {self.file_path2}")
            threading.Thread(target=self.record_bag_file, args=(self.file_path1, self.file_path2)).start()
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.record_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            print("Recording stopped.")
            if self.pipeline1:
                self.pipeline1.stop()
            if self.pipeline2:
                self.pipeline2.stop()
    
    def select_path1(self):
        self.file_path1 = filedialog.asksaveasfilename(defaultextension=".bag", 
                                                      filetypes=[("Bag files", "*.bag"), ("All files", "*.*")])
        if self.file_path1:
            self.path_label1.config(text=f"Path: {self.file_path1}")
        else:
            self.path_label1.config(text="No path selected for Stream 1")

    def select_path2(self):
        self.file_path2 = filedialog.asksaveasfilename(defaultextension=".bag", 
                                                      filetypes=[("Bag files", "*.bag"), ("All files", "*.*")])
        if self.file_path2:
            self.path_label2.config(text=f"Path: {self.file_path2}")
        else:
            self.path_label2.config(text="No path selected for Stream 2")

    def record_bag_file(self, output_path1, output_path2):
        ctx = rs.context()
        devices = ctx.query_devices()
        dev0 = ctx.query_devices()[0]  
        dev1 = ctx.query_devices()[1]

        # Configure both streams
        self.pipeline1 = rs.pipeline()
        config1 = rs.config()

        self.pipeline2 = rs.pipeline()
        config2 = rs.config()

        config1.enable_device(dev0.get_info(rs.camera_info.serial_number))
        config1.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.color_fps)
        config1.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps)
        config1.enable_record_to_file(output_path1)

        config2.enable_device(dev1.get_info(rs.camera_info.serial_number))
        config2.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.color_fps)
        config2.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps)
        config2.enable_record_to_file(output_path2)

        # Start streaming for both streams
        profile1 = self.pipeline1.start(config1)
        profile2 = self.pipeline2.start(config2)
        depth_sensor1 = profile1.get_device().first_depth_sensor()
        depth_sensor2 = profile2.get_device().first_depth_sensor()

        # Using preset MediumDensity for recording
        # MediumDensity : Balance between Fill factor and accuracy
        # Others : Custom = 0, Default = 1, Hand = 2, HighAccuracy = 3, HighDensity = 4, MediumDensity = 5
        depth_sensor1.set_option(rs.option.visual_preset, 5)
        depth_sensor2.set_option(rs.option.visual_preset, 5)
        
        # Set auto exposure for both streams
        depth_sensor1 = profile1.get_device().query_sensors()[0]
        color_sensor1 = profile1.get_device().query_sensors()[1]
        depth_sensor1.set_option(rs.option.enable_auto_exposure, self.auto_exposure)
        color_sensor1.set_option(rs.option.enable_auto_exposure, self.auto_exposure)

        depth_sensor2 = profile2.get_device().query_sensors()[0]
        color_sensor2 = profile2.get_device().query_sensors()[1]
        depth_sensor2.set_option(rs.option.enable_auto_exposure, self.auto_exposure)
        color_sensor2.set_option(rs.option.enable_auto_exposure, self.auto_exposure)

        # Mannually set the exposure : .set_option(rs.option.exposure, exposure)

        """ # Load the JSON settings file
        # => Both recording and loading the file does not work
        # => https://github.com/IntelRealSense/librealsense/issues/10184
        path_to_json = "./record_preset.json"
        jsonObj = json.load(open("./record_preset.json"))
        json_string= str(jsonObj).replace("'", '\"')
        device = profile.get_device()  # Get the device
        advanced_mode = rs.rs400_advanced_mode(device)
        advanced_mode.load_json(json_string) """

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()


# TODO : add laser