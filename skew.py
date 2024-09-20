import cv2
import numpy as np

# Load the image
image = cv2.imread('tes.jpg')

# Get image dimensions
(h, w) = image.shape[:2]

# Define the skew matrix
# Adjust the skew factors as needed
skew_factor = 0.5
M = np.float32([[1, skew_factor, 0], [1, 0, 0]])

# Apply the skew transformation
skewed_image = cv2.warpAffine(image, M, (w, h))

# Display the original and skewed images
cv2.imshow('Original Image', image)
cv2.imshow('Skewed Image', skewed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
