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
        
        # Path selector button
        self.path_button = tk.Button(root, text="Select Path", command=self.select_path)
        self.path_button.pack(pady=10)
        
        # Label to display the selected path
        self.path_label = tk.Label(root, text="No path selected")
        self.path_label.pack(pady=10)
        
        self.recording = False
        self.file_path = ""
        self.pipeline = None

    def start_recording(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a path to save the file.")
            return
        
        if not self.recording:
            self.recording = True
            self.record_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            print(f"Recording started, saving to {self.file_path}")
            threading.Thread(target=self.record_bag_file, args=(self.file_path,)).start()
        else:
            messagebox.showinfo("Info", "Recording is already in progress.")
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.record_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            print("Recording stopped.")
            if self.pipeline:
                self.pipeline.stop()
        else:
            messagebox.showinfo("Info", "No recording is in progress.")
    
    def select_path(self):
        self.file_path = filedialog.asksaveasfilename(defaultextension=".bag", 
                                                      filetypes=[("Bag files", "*.bag"), ("All files", "*.*")])
        if self.file_path:
            self.path_label.config(text=f"Path: {self.file_path}")
        else:
            self.path_label.config(text="No path selected")

    def record_bag_file(self, output_path):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Configure depth and color streams
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.color_fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps)
        config.enable_record_to_file(output_path)

        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        # Using preset MediumDensity for recording
        # MediumDensity : Balance between Fill factor and accuracy
        # Others : Custom = 0, Default = 1, Hand = 2, HighAccuracy = 3, HighDensity = 4, MediumDensity = 5
        depth_sensor.set_option(rs.option.visual_preset, 5)
        
        # Set auto exposure
        depth_sensor = profile.get_device().query_sensors()[0]
        color_sensor = profile.get_device().query_sensors()[1]
        depth_sensor.set_option(rs.option.enable_auto_exposure, self.auto_exposure)
        color_sensor.set_option(rs.option.enable_auto_exposure, self.auto_exposure)

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