import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

baslar_file = select_file()
cap = cv2.VideoCapture(baslar_file)
wait_key = 1

# Setup the aruco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
aruco_params = cv2.aruco.DetectorParameters()

aruco_params.errorCorrectionRate = 0.2
aruco_params.polygonalApproxAccuracyRate = 0.05
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
# aruco_params.minMarkerPerimeterRate = 0.01

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

while cap.isOpened():
    ret, image = cap.read()

    if ret:
        # Detect the aruco markers
        corners, ids, rejected = detector.detectMarkers(image)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', image)
        cv2.resizeWindow('Frame', 864, 648)
        
        key = cv2.waitKey(wait_key)

        # Press esc close the image window
        if key == 27:
            break

        # Press p to view the video frame by frame
        if key == ord('p'):
            wait_key = 0 if wait_key == 1 else 1
    else:
        break

cap.release()
cv2.destroyAllWindows()