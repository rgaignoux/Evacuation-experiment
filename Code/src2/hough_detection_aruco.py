# 1. Extract white pixels
# 2. Apply Hough Line Transform
# 3. Find two lines that are parralel (or close), and of the same length (or close)
# 4. Apply motion deblur

import cv2
import numpy as np

path = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\images\\color_images\\images\\color_frame_811.png"
image = cv2.imread(path)

# 1. Extract white pixels
white = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
edges = cv2.Canny(white, 50, 150, apertureSize=3)

# 2. Apply Hough Line Transform
angles = {}
lengths = {}
lines = {}

lines_probabilistic = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
image_probabilistic = image.copy()
if lines_probabilistic is not None:
    count = 0
    for x1, y1, x2, y2 in lines_probabilistic[:, 0]:
        angle = np.arctan((y2 - y1) / (x1 - x2)) * 180 / np.pi
        angles[count] = angle

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lengths[count] = length

        lines[count] = (x1, y1, x2, y2)
        count += 1
        cv2.line(image_probabilistic, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Probabilistic Hough Line Transform', image_probabilistic)
cv2.waitKey(0)
cv2.destroyAllWindows()

parallel_lines = []
parallel_lines_angles = []
image_probabilistic = image.copy()

# 3. Find two lines that are parralel (or close), and of the same length (or close)
for key, value in angles.items():
    for key2, value2 in angles.items():
        if key == key2:
            continue
        if abs(value - value2) < 5:
            if abs(lengths[key] - lengths[key2]) < 5:
                print("Angles : {} ; {}".format(angles[key], angles[key2]))
                line = lines[key]
                parallel_lines.append(line)

                cv2.line(image_probabilistic, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
                parallel_lines_angles.append(angles[key])
                parallel_lines_angles.append(angles[key2])

# Remove outliers with Z-score method
mean_angle1 = np.mean(parallel_lines_angles) * 180 / np.pi

parallel_lines_angles = np.array(parallel_lines_angles)
z_scores = np.abs((parallel_lines_angles - np.mean(parallel_lines_angles)) / np.std(parallel_lines_angles))
threshold = 4
parallel_lines_angles = parallel_lines_angles[z_scores < threshold]

mean_angle2 = np.mean(parallel_lines_angles)

print("Mean angle 1: {} ; Mean angle 2: {}".format(mean_angle1, mean_angle2))

# 4. Apply motion deblur

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = np.float32(grayscale)/255.0
IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

ang = 180 - mean_angle2
ang = np.deg2rad(ang)
d = 10
SNR = 15
print("Angle: {} ; d: {} ; SNR: {}".format(ang, d, SNR))

def motion_deblur(ang, d, SNR, img, IMG):
    noise = 10**(-0.1*SNR)
    psf = motion_kernel(ang, d)
    psf /= psf.sum()
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
    RES = cv2.mulSpectrums(IMG, iPSF, 0)
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res = np.roll(res, -kh//2, 0)
    res = np.roll(res, -kw//2, 1)

    return res

deblured = motion_deblur(ang, d, SNR, img, IMG)

# Apply gaussian blur
deblured = cv2.GaussianBlur(deblured, (5, 5), 0)

cv2.imshow('Motion deblur', deblured)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Threshold image

# Where pixels are < 150, set to 0
res = np.uint8(deblured * 255)
mask = res <= 150
res[mask] = 0

cv2.imshow('Threshold', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Aruco markers detection

# Setup the aruco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
aruco_params = cv2.aruco.DetectorParameters()

aruco_params.errorCorrectionRate = 0.2 # default 0.6
aruco_params.polygonalApproxAccuracyRate = 0.05 # default 0.03

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

corners, ids, rejected = detector.detectMarkers(res)
output_image = res.copy()
cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

cv2.imshow('Aruco', output_image)

if ids is not None:
    print("Found Aruco marker with ID: ", ids[0][0])
    marker_id = ids[0][0]
    print("Marker ID: ", marker_id)
    marker_size = 200  # Taille de l'image générée
    aruco_marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imshow("Marker", aruco_marker_image)

cv2.waitKey(0)
cv2.destroyAllWindows()