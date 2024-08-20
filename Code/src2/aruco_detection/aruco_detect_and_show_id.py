import cv2
import numpy as np

path = "deblurred.png"
img = cv2.imread(path, cv2.IMREAD_COLOR)

# Setup the aruco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

corners, ids, rejected = detector.detectMarkers(img)
output_image = img.copy()
cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

cv2.imshow('Aruco', output_image)

if ids is None:
    print("No Aruco marker found")

if ids is not None:
    for marker_id in ids:
        print("Found Aruco marker with ID: ", marker_id[0])
        id = marker_id[0]
        print("Marker ID: ", marker_id)
        marker_size = 200
        aruco_marker_image = cv2.aruco.generateImageMarker(aruco_dict, id, marker_size)
        cv2.imshow("Marker", aruco_marker_image)

cv2.waitKey(0)
cv2.destroyAllWindows()