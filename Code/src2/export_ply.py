import numpy as np                     
import pyrealsense2 as rs             
import numpy as np
import cv2
from post_process import post_process_depth_frame

pc = rs.pointcloud()
pipe = rs.pipeline()

#Create a config and configure the pipeline to stream

config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipe.start(config)
colorizer = rs.colorizer()

for x in range(10):
	pipe.wait_for_frames()


try:
    while True:
        frames = pipe.wait_for_frames()

        # Get aligned frames
        depth_frame = frames.get_depth_frame() 
        color_frame = frames.get_color_frame()

        depth_frame = post_process_depth_frame(depth_frame)

        depth_frame_colorized = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_frame_colorized.get_data())
        cv2.imshow('Depth', depth_image)
        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord('s'):
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            print("Saving ply...")
            points.export_to_ply("2.ply", color_frame)

finally:
    pipe.stop()
    cv2.destroyAllWindows()






print("Done")	
