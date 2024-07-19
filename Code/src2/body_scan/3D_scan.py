import pyrealsense2 as rs
import cv2
import numpy as np
from post_process import post_process_depth_frame

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = pipeline.start(config)
colorizer = rs.colorizer()
pc = rs.pointcloud()
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()

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
            points.export_to_ply("1.ply", color_frame)
    
finally:
    pipeline.stop()
    cv2.destroyAllWindows()




