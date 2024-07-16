import cv2
import numpy as np

# 1. Load image
image = cv2.imread('depth_image.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Depth Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Blurred Depth Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Apply Mean Shift algorithm
# Convert grayscale image to a 3-channel image for Mean Shift
image_3channel = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

# Apply pyrMeanShiftFiltering
spatial_radius = 21  # Spatial window radius
color_radius = 51  # Color window radius
max_level = 1  # Maximum level of the pyramid
mean_shift_result = cv2.pyrMeanShiftFiltering(image_3channel, spatial_radius, color_radius, max_level)

# Convert back to grayscale
mean_shift_result_gray = cv2.cvtColor(mean_shift_result, cv2.COLOR_BGR2GRAY)

# 4. Display the result
cv2.imshow('Mean Shift Result', mean_shift_result_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get unique intensities from the result image
unique_intensities = np.unique(mean_shift_result)
print(len(unique_intensities))

# Display each cluster in separate windows based on intensity
for intensity in unique_intensities:
    # Create a mask for pixels with the current intensity
    mask = np.uint8(mean_shift_result_gray == intensity)
    
    # Apply the mask to the original image to show only pixels with current intensity
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Display the result in a separate window
    cv2.imshow(f'Cluster Intensity {intensity}', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()