import cv2
import numpy as np

def sharpen_image_basic(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sharpen_image_strong(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(image):
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def laplacian_sharpen(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def high_pass_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

def clahe_sharpen(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def canny_edge_detection(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 100, 200)
    return edges

def main(image_path):
    image = cv2.imread(image_path)

    basic_sharpened = sharpen_image_basic(image)
    strong_sharpened = sharpen_image_strong(image)
    unsharp_masked = unsharp_mask(image)
    laplacian_sharpened = laplacian_sharpen(image)
    high_pass_sharpened = high_pass_filter(image)
    clahe_sharpened = clahe_sharpen(image)

    basic_edges = canny_edge_detection(basic_sharpened)
    strong_edges = canny_edge_detection(strong_sharpened)
    unsharp_edges = canny_edge_detection(unsharp_masked)
    laplacian_edges = canny_edge_detection(laplacian_sharpened)
    high_pass_edges = canny_edge_detection(high_pass_sharpened)
    clahe_edges = canny_edge_detection(clahe_sharpened)

    cv2.imshow('Basic Sharpened', basic_sharpened)
    cv2.imshow('Basic Edges', basic_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Strong Sharpened', strong_sharpened)
    cv2.imshow('Strong Edges', strong_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Unsharp Mask', unsharp_masked)
    cv2.imshow('Unsharp Edges', unsharp_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Laplacian Sharpened', laplacian_sharpened)
    cv2.imshow('Laplacian Edges', laplacian_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('High Pass Filter', high_pass_sharpened)
    cv2.imshow('High Pass Edges', high_pass_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('CLAHE Sharpened', clahe_sharpened)
    cv2.imshow('CLAHE Edges', clahe_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main('deblurred.png')
