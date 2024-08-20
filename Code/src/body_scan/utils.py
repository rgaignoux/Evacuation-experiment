import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation as R
from concave_hull import concave_hull, concave_hull_indexes

def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
    return file_path

# Red : x-axis
# Green : y-axis
# Blue : z-axis
def visualize(pcd):
    viewer = o3d.visualization.VisualizerWithEditing()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    print(viewer.get_picked_points())


def project_to_xy_plane(pcd):
    # Get points from the point cloud
    points = np.asarray(pcd.points)
    
    # Set Z coordinate to 0
    points[:, 2] = 0
    
    # Update point cloud with new points
    pcd.points = o3d.utility.Vector3dVector(points)

    """ # Remove outliers
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
    pcd = pcd.select_by_index(ind) """
    
    return pcd


def project_to_xz_plane(pcd):
    # Get points from the point cloud
    points = np.asarray(pcd.points)
    
    # Set Y coordinate to 0
    points[:, 1] = 0
    
    # Update point cloud with new points
    pcd.points = o3d.utility.Vector3dVector(points)

    """ # Remove outliers
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
    pcd = pcd.select_by_index(ind) """
    
    return pcd


def get_body_contour(points_2d, save_path="body_projection.png"):
    # Slider update function
    def update(val):
        length_threshold = slider.val
        idxes = concave_hull_indexes(points_2d, length_threshold=length_threshold)
        vertices = points_2d[idxes]

        ax.clear()
        ax.plot(points_2d[:, 0], points_2d[:, 1], '.', label='Points')
        ax.plot(vertices[:, 0], vertices[:, 1], 'r-', label='Body contour', linewidth=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.axis('equal')
        ax.legend()
        fig.canvas.draw_idle()

        # Compute X and Z lengths
        x_length = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        z_length = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        
        print("X length:", x_length)
        print("Z length:", z_length)
        
        # Save the image after each update
        fig.savefig(save_path)
        print(f"Image saved to {save_path}")

    # Create tkinter window to create a slider
    root = tk.Tk()
    root.title("Concave Hull Slider")

    # Initial plot
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    ax.plot(points_2d[:, 0], points_2d[:, 1], '.', label='Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.axis('equal')
    ax.legend()

    # Add margin at the bottom for the slider
    plt.subplots_adjust(bottom=0.25)

    # Create the slider
    slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Length Threshold', 0.01, 1.0, valinit=0.15, valstep=0.01)
    slider.on_changed(update)

    canvas.draw()
    root.mainloop()


# Camera settings
width = 640
height = 480
fps = 15

def create_pipeline(serial_number):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Set emitter enabled to "laser"
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Use Medium Density preset
    depth_sensor.set_option(rs.option.visual_preset, 5)

    # Set manual exposure to 5000
    depth_sensor.set_option(rs.option.exposure, 8500)

    return pipeline

def create_pipeline_IR(serial_number, width, height, fps):
    # Create a pipeline
    print("Creating pipeline for camera with serial number", serial_number, "...")
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Set emitter_enabled to 'laser'
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Use Medium Density preset
    depth_sensor.set_option(rs.option.visual_preset, 5)

    # Set manual exposure to 8500
    depth_sensor.set_option(rs.option.exposure, 8500)

    # Get instrinsics
    instrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

    print("Done")
    return pipeline, instrinsics


def save_IR(pipe, path_IR):
    try:
        frames = pipe.wait_for_frames()
        # Save IR
        infrared_frame = frames.get_infrared_frame()
        infrared_image = np.asanyarray(infrared_frame.get_data())
        print("Saving IR to", path_IR)
        cv2.imwrite(path_IR, infrared_image)
        print("Done")
    finally:
        pipe.stop() 


def save_ply(path, pipe):
    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = post_process_depth_frame(depth_frame, temporal_smooth_alpha=0.2, temporal_smooth_delta=40)

        # Create save_to_ply object
        ply = rs.save_to_ply(path)
        print("Saving PLY to ", path)
        ply.process(depth_frame)
        print("Done")
    finally:
        pipe.stop() 


def post_process_depth_frame(depth_frame, min_distance=0, max_distance=4, decimation_magnitude = 1.0, spatial_magnitude = 2.0, spatial_smooth_alpha = 0.5, spatial_smooth_delta = 20, temporal_smooth_alpha = 0.4, temporal_smooth_delta = 20, fill_hole = False):
    """
    Apply post processing filters to the depth frame of a RealSense camera.
    More information about the filters can be found at:
    - https://dev.intelrealsense.com/docs/post-processing-filters
    - https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
    - https://intelrealsense.github.io/librealsense/doxygen/namespacers2.html
    """
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters
    decimation_filter = rs.decimation_filter()
    threshold_filter = rs.threshold_filter()
    depth_to_disparity = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(1) # https://intelrealsense.github.io/librealsense/doxygen/classrs2_1_1hole__filling__filter.html

    # Apply the control parameters for the filters
    decimation_filter.set_option(rs.option.filter_magnitude, decimation_magnitude)
    threshold_filter.set_option(rs.option.min_distance, min_distance)
    threshold_filter.set_option(rs.option.max_distance, max_distance)
    spatial_filter.set_option(rs.option.filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(rs.option.filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(rs.option.filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(rs.option.filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    # Post processing order : https://dev.intelrealsense.com/docs/post-processing-filters
    # Depth Frame >> Decimation Filter >> Depth2Disparity Transform >> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform >> Hole Filling Filter >> Filtered Depth
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = threshold_filter.process(filtered_frame)
    filtered_frame = depth_to_disparity.process(filtered_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)
    filtered_frame = disparity_to_depth.process(filtered_frame)
    if fill_hole:
        filtered_frame = hole_filling.process(filtered_frame)
    
    # Cast to depth_frame so that we can use the get_distance method afterwards
    depth_frame_filtered = filtered_frame.as_depth_frame()

    return depth_frame_filtered


def get_euler_angles(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rotation_matrix)
    euler_angles_rad = r.as_euler('xyz', degrees=False)
    euler_angles_deg = np.degrees(euler_angles_rad)
    return euler_angles_deg


def get_transform_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def get_inverse_homogenous(R, t):
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -(R.T @ t).flatten()
    return T


def draw_axes(img, origin, xyz_axes):
    origin = tuple(origin.ravel().astype(int))
    x = tuple(xyz_axes[0].ravel().astype(int))
    y = tuple(xyz_axes[1].ravel().astype(int))
    z = tuple(xyz_axes[2].ravel().astype(int))

    img = cv2.line(img, origin, x, (255,0,0), 5)
    img = cv2.line(img, origin, y, (0,255,0), 5)
    img = cv2.line(img, origin, z, (0,0,255), 5)
    return img


def visualize_axes(img, origin, rvec, tvec, cameraMatrix, distCoeffs):
    # Project the axes
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    xyz_axes, _ = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
    img = draw_axes(img, origin, xyz_axes)
    return img


def draw_corners_and_axes(img, corners, rvec, tvec, cameraMatrix, distCoeffs):
    # Draw the corners
    img = cv2.drawChessboardCorners(img, (9,6), corners, True)

    # Draw the axes
    origin = corners[0].ravel()
    img = visualize_axes(img, origin, rvec, tvec, cameraMatrix, distCoeffs)
    return img


def get_extrinsics(path_IR, pattern_size, cameraMatrix, distCoeffs):
    img = cv2.imread(path_IR, cv2.IMREAD_GRAYSCALE)

    # Chessboard points in object coordinate system
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    # Find chessboard corners (findChessboardCornersSB gives better results than findChessboardCorners)
    ret, corners = cv2.findChessboardCornersSB(img, pattern_size, flags = cv2.CALIB_CB_EXHAUSTIVE)

    if not ret:
        print("Chessboard corners not found!")
        return None
    
    # Compute the extrinsics parameters using solvePnP
    _, rvec, tvec = cv2.solvePnP(objp, corners, cameraMatrix, distCoeffs)

    # Draw the corners and the axes
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color = draw_corners_and_axes(img_color, corners, rvec, tvec, cameraMatrix, distCoeffs)

    return rvec, tvec, img_color
    

def get_instrinsics_matrix(intrinsics):
    fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    cameraMatrix = np.array([
        [fx, 0,  ppx],
        [0,  fy, ppy],
        [0,  0,  1]
    ])
    distCoeffs = np.array([0,0,0,0,0], dtype=np.float32)

    return cameraMatrix, distCoeffs