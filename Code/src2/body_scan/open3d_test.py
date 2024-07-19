import open3d
import numpy as np

# Load the mesh
path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\3D_models\\box_right3.ply"
mesh = open3d.io.read_triangle_mesh(path)

# Visualize the original mesh
viewer = open3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(mesh)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()

entire_bbox = mesh.get_axis_aligned_bounding_box()
print(entire_bbox)