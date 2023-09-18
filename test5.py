import cv2
import numpy as np
import os

# Set the path to the Downloads folder and the name of the image file
# downloads_folder = os.path.expanduser("~/Downloads")
# img_file = "testben1.jpg"

# Load the image
img_path =  "./images/Metal_1.jpg"
img = cv2.imread(img_path)

# Halve the image size
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set the threshold for "black"
black_threshold = 25

# Create a mask of everything that is below the black threshold
mask = gray < black_threshold

# Create a new image that is completely white
white_img = np.full_like(img, (255, 255, 255))

# Apply the mask to the white image, making only the black pixels stay black and replacing all other pixels with white
result = np.where(mask[..., None], white_img, img)

# Find contours around the black holes in the image
_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours around the black holes in the image
num_circles = 0
for contour in contours:
    # Approximate the contour as a polygon
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    # If the polygon has enough vertices (at least 5), consider it a circle and draw it
    if len(approx) >= 5:
        num_circles += 1
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(result, ellipse, (0, 255, 0), 2)

# Show the result and print the number of circles drawn
print(f"Number of circles drawn: {num_circles}")
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()