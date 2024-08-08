import utils
import cv2
import open3d as o3d
import pyrealsense2 as rs
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def save_ply_and_IR(pipe, path_ply, path_IR):
    try:
        frames = pipe.wait_for_frames()
        
        # Save IR
        infrared_frame = frames.get_infrared_frame()
        infrared_image = np.asanyarray(infrared_frame.get_data())
        print("Saving IR to", path_IR)
        cv2.imwrite(path_IR, infrared_image)
        print("Done")

        # Save PLY
        depth_frame = frames.get_depth_frame()
        depth_frame = utils.post_process_depth_frame(depth_frame, temporal_smooth_alpha=0.2, temporal_smooth_delta=40)
        ply = rs.save_to_ply(path_ply)
        print("Saving PLY to ", path_ply)
        ply.process(depth_frame)
        print("Done")
    finally:
        pipe.stop() 


def get_ground(pcd):
    # Select the 2 corners of the rectange
    print("Please select 2 points on the ground (hold shift + left click)")
    points_ground = utils.visualize(pcd)

    while len(points_ground) != 2:
        print("Please select exactly 2 points")
        points_ground = utils.visualize(pcd)
    
    min_bound = points_ground.min(axis=0)
    max_bound = points_ground.max(axis=0)

    # Add some margin
    min_bound[1] -= 0.1 
    max_bound[1] += 0.1

    # Extract the ground
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    ground = pcd.crop(bbox)

    # Remove outliers
    cl, ind = ground.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ground = ground.select_by_index(ind)

    return ground

def get_angle(pcd, plot = True):
    ground = get_ground(pcd)
    
    # Project all points on YZ plane
    points = np.asarray(ground.points)
    points = points[:, [1, 2]]

    # Fit a line to the points with linear regression
    z = points[:, 1].reshape(-1, 1)
    y = points[:, 0]
    model = LinearRegression()
    model.fit(z, y)

    # Coefficients of the computed line
    a = model.coef_[0]

    if (plot):
        plt.scatter(z, y, color='blue')
        plt.plot(z, model.predict(z), color='red', linewidth=2)
        plt.xlabel('Z')
        plt.ylabel('Y')
        plt.show()
        
    # Angle of the ground
    angle = np.arctan(a) * 180 / np.pi

    return angle

# Front angle
front_pcd = o3d.io.read_point_cloud("front.ply")
angle_front = get_angle(front_pcd)

# Back angle
back_pcd = o3d.io.read_point_cloud("back.ply")
angle_back = get_angle(back_pcd)

print("Front angle: ", angle_front)
print("Back angle: ", angle_back)