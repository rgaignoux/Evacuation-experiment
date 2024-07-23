import open3d as o3d
import numpy as np

# Translation / Rotation values
distance_cameras = -2.6
angles_cameras = -24

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

def crop_body(pcd_front, pcd_back, distance_cameras, angle_cameras):
    # Rotate front pcd by 180 degrees on Y axis
    R = pcd_front.get_rotation_matrix_from_xyz((0, np.pi, 0))
    pcd_front.rotate(R, center=(0, 0, 0))

    # Rotate both pcds by camera_angles degrees on X axis
    rad = angle_cameras * np.pi / 180
    R = pcd_front.get_rotation_matrix_from_xyz((-rad, 0, 0))
    pcd_front.rotate(R, center=(0, 0, 0))

    R = pcd_back.get_rotation_matrix_from_xyz((rad, 0, 0))
    pcd_back.rotate(R, center=(0, 0, 0))

    # Translate front pcd by distance_between_cameras on Z axis
    pcd_front.translate((0, 0, distance_cameras))

    # Merge both meshes
    pcd = pcd_front + pcd_back

    # Get the body bounding box
    entire_bbox = pcd.get_axis_aligned_bounding_box()

    body_center = np.array([0, 0, -1.2])
    center_bbox_min = body_center - np.array([0.5, 0, 0.5])
    center_bbox_max = body_center + np.array([0.5, 0, 0.5])
    center_bbox_min[1] = entire_bbox.min_bound[1]
    center_bbox_max[1] = entire_bbox.max_bound[1]

    # Crop the body
    center_bbox = o3d.geometry.AxisAlignedBoundingBox(center_bbox_min, center_bbox_max)
    body = pcd.crop(center_bbox)

    return body

# Load front point cloud
pcd_front = o3d.io.read_point_cloud("C:\\Users\\Robin\\Documents\\Stage2024\\Code\\3D_models\\front1.ply")
pcd_front = pcd_front.uniform_down_sample(every_k_points=5)

# Load back point cloud
pcd_back = o3d.io.read_point_cloud("C:\\Users\\Robin\\Documents\\Stage2024\\Code\\3D_models\\back1.ply")
pcd_back = pcd_back.uniform_down_sample(every_k_points=5)

# Remove outliers and visualize
cl, ind = pcd_front.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_front = pcd_front.select_by_index(ind)
visualize(pcd_front)

cl, ind = pcd_back.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_back = pcd_back.select_by_index(ind)
visualize(pcd_back)

body = crop_body(pcd_front, pcd_back, distance_cameras, angles_cameras)
visualize(body)








