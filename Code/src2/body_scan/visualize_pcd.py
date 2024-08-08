import open3d as o3d
import utils

path = utils.select_file()
pcd = o3d.io.read_point_cloud(path)
utils.visualize(pcd)