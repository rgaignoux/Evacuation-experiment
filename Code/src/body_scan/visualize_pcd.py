import open3d as o3d
import utils

path = utils.select_file()
pcd = o3d.io.read_point_cloud(path)
utils.visualize(pcd)

# Crop
bbox_min = [-1, -0.5, -2.7]
bbox_max = [1.5, -1, -2]

bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
cropped = pcd.crop(bbox)

utils.visualize(cropped)