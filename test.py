import cv2
import numpy as np
import time
import os


# load image from file
image_path = "images/blobs.jpg"
img = cv2.imread(image_path)

# define black color range
lower_black = np.array([0, 0, 0])
upper_black = np.array([50, 50, 50])

# initialize list to store hole areas
hole_areas = []

# convert to grayscale and apply thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
temp, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# apply black color mask to thresholded image
mask = cv2.inRange(img, lower_black, upper_black)
masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

# find contours in masked image
contours,_  = cv2.findContours(masked_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# loop through contours and filter for black hole shapes
hole_count = 0
for cnt in contours:
    # check if contour has enough points to approximate a curve
    if len(cnt) >= 5:
        # fit an ellipse to the contour
        ellipse = cv2.fitEllipse(cnt)
        # calculate aspect ratio of ellipse
        aspect_ratio = min(ellipse[1])/max(ellipse[1])
        # check if aspect ratio is close to 1 to filter for circular shapes
        if 0.7 <= aspect_ratio <= 1.3:
            hole_areas.append(np.pi(ellipse[1][0]/2)(ellipse[1][1]/2))
            hole_count += 1
            # draw ellipse on original image
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# check if 7 holes have been detected
time.sleep(25)
if hole_count != 0:
    print("All holes detected.")
else:
    print("No Holes Detected")

# display image and wait for key press
cv2.imshow("image", img)
cv2.waitKey(0)

# release window and print hole areas
cv2.destroyAllWindows()
print("Hole areas:", hole_areas)
