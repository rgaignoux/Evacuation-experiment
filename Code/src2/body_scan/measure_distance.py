import pyrealsense2 as rs
import cv2
import numpy as np
import utils

def select_point(event, x, y, flags, param):
    image, points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', image)

def pick_points(image):
    points = []
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', select_point, [image, points])
    while True:
        cv2.imshow('image', image)
        if len(points) == 2:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return points

# Read the bag file
bag_file = utils.select_file()

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False) # False: no frame drop

wait_key = 1
pick = False

try:
    while True:
        frames = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = utils.post_process_depth_frame(depth_frame, fill_hole=True)
        colorizer = rs.colorizer()
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.imshow('Depth Image', depth_color_image)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Color Image', color_image)

        if(pick):
            # Pick two points
            points = pick_points(color_image)

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            u = points[0]
            v = points[1]

            udist = depth_frame.get_distance(u[0], u[1])
            vdist = depth_frame.get_distance(v[0], v[1])

            point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [u[0], u[1]], udist)
            point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [v[0], v[1]], vdist)

            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            print("Distance between points: ", distance)

            pick = False

        key = cv2.waitKey(wait_key)

        if key == ord('p'):
            pick = not pick

        if key == ord('q'):
            break

        if key == ord('s'):
            cv2.imwrite('image.png', color_image)

        if key == ord('d'):
            wait_key = 0 if wait_key == 1 else 1

finally:
    pipe.stop()
    cv2.destroyAllWindows()