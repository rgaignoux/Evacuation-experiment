import pyrealsense2 as rs
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def find_regional_minima(depth_frame, region_size = 5):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    
    # Liste pour stocker les minima régionaux
    regional_minima = []
    
    # Parcourir l'image par régions
    for i in range(0, width, region_size):
        for j in range(0, height, region_size):
            # Initialiser le minimum pour la région actuelle
            min_distance = float('inf')
            min_position = (i, j)
            
            # Parcourir les pixels de la région actuelle
            for x in range(i, min(i + region_size, width)):
                for y in range(j, min(j + region_size, height)):
                    distance = depth_frame.get_distance(x, y)
                    
                    # Ignorer les distances nulles (pas de données)
                    if distance != 0.0 and distance < min_distance:
                        min_distance = distance
                        min_position = (x, y)
            
            # Ajouter le minimum régional à la liste
            if min_distance != float('inf'):
                regional_minima.append((min_position[0], min_position[1], min_distance))
    
    return regional_minima


def select_file():
    root = tk.Tk()
    root.withdraw()  # Fermer la fenêtre principale
    file_path = filedialog.askopenfilename()  # Ouvrir la fenêtre de dialogue pour sélectionner un fichier
    return file_path

# Read the bag file
bag_file = select_file()
bag_file = bag_file.replace('\\', '\\\\')

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False) # False: no frame drop

try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Find the regional minima
        regional_minima = find_regional_minima(depth_frame)
        print("Regional minima: {}".format(regional_minima))

        colorizer = rs.colorizer(2)
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Draw the regional minima
        for x, y, distance in regional_minima:
            cv2.circle(depth_color_image, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Depth Image', depth_color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()



