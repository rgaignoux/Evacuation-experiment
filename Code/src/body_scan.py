import pyrealsense2 as rs
import numpy as np
import time
from post_process import post_process_depth_frame

width = 640
height = 480
fps = 30

def create_pipeline(serial_number):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start streaming
    profile = pipeline.start(config)

    return pipeline


def save_ply(path, pipe):
    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        # Create save_to_ply object
        ply = rs.save_to_ply(path)

        print("Saving PLY to ", path)
        # Apply the processing block to the frameset which contains the depth frame and the texture
        ply.process(colorized)
        print("Done")
    finally:
        pipe.stop()


right_pipe = create_pipeline("")
left_pipe = create_pipeline("")

# Wait for auto-exposure to stabilize
for x in range(10):
    right_pipe.wait_for_frames()
    left_pipe.wait_for_frames()

# Wait 5 seconds
for i in range(5):
    print("Capture will start in ", 5-i, " seconds")
    time.sleep(1)

# Save the PLY files
save_ply("right.ply", right_pipe)
save_ply("left.ply", left_pipe)









