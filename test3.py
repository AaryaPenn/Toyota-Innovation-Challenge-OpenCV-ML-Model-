import cv2
import numpy as np
import os

# # get path to user's Downloads directory
# downloads_path = os.path.expanduser("~") + "/Downloads"

# # load image from file
# image_path = os.path.join(downloads_path, "testben1.jpg")
img = cv2.imread("./images/Metal_1.jpg")

# denoise image using fastNlMeansDenoising
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# define black color range
lower_black = np.array([0, 0, 0])
upper_black = np.array([70, 70, 70])

# initialize list to store hole areas
hole_areas = []

# convert to grayscale and apply thresholding
gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

# apply black color mask to thresholded image
mask = cv2.inRange(denoised, lower_black, upper_black)
masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

# find contours in masked image
contours, hierarchy = cv2.findContours(masked_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
            hole_areas.append(area)
            hole_count += 1
            # draw ellipse on original image
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# check if holes have been detected
if hole_count == 0:
    print("No holes detected.")
elif hole_count == 7:
    print("All holes detected.")
else:
    print("Detected", hole_count, "holes.")

# display image and wait for key press
img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
thresh = cv2.resize(thresh, (0, 0), fx=0.4, fy=0.4)
gray = cv2.resize(gray, (0, 0), fx=0.4, fy=0.4)

cv2.imshow("image", img)
cv2.imshow("image2", thresh)
cv2.imshow("image3", gray)

cv2.waitKey(0)

# release window and print hole areas
cv2.destroyAllWindows()
print("Hole areas:", hole_areas)