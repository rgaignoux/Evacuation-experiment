import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

path = "C:\\Users\\Robin\\Documents\\Stage2024\\Dataset\\aruco_images\\perspective_projection_test.png"
img = cv2.imread(path)
# Coordinates that you want to Perspective Transform
pts1 = np.float32([[115,19],[192,36],[95,89],[174,107]])
# Size of the Transformed Image
pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (400,400))
cv2.imshow('Image', dst)

################################
# Attempt to enhance the image #
################################

# Where pixels are < 200, set to 0
mask = dst <= 200
dst[mask] = 0

# Add with border to the image of size 20 px
dst = cv2.copyMakeBorder(dst,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])

# Setup the aruco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
aruco_params = cv2.aruco.DetectorParameters()

aruco_params.errorCorrectionRate = 0.2 # default 0.6
aruco_params.polygonalApproxAccuracyRate = 0.05 # default 0.03

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

corners, ids, rejected = detector.detectMarkers(dst)
output_image = dst.copy()
cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

cv2.imshow('Aruco', output_image)

if ids is not None:
    print("Found Aruco marker with ID: ", ids[0][0])
    marker_id = ids[0][0]
    print("Marker ID: ", marker_id)
    marker_size = 200 
    aruco_marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imshow("Marker", aruco_marker_image)

cv2.waitKey(0)
cv2.destroyAllWindows()