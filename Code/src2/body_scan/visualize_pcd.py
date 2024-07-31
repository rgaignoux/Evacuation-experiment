import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog

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


def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la feqqnêtre de dialogue pour sélectionner un fichier
    return file_path

# Read the file
path = select_file()
pcd_front = o3d.io.read_point_cloud(path)

cl, ind = pcd_front.remove_radius_outlier(nb_points=16, radius=0.05)
pcd_front = pcd_front.select_by_index(ind)
visualize(pcd_front)