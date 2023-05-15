import cv2
import numpy as np

# Load image
lower_black = np.array([0, 0, 0])
upper_black = np.array([70, 70, 70])

image = cv2.imread('./images/Metal_1.jpg')
denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
mask = cv2.inRange(denoised, lower_black, upper_black)
masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
# contours, hierarchy = cv2.findContours(masked_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 40
params.maxArea = 200

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 0.9

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.2
	
# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
contours, hierarchy = cv2.findContours(masked_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# loop through contours and filter for black hole shapes
hole_areas = []
hole_count = 0
for cnt in contours:
    # check if contour has enough points to approximate a curve
    if len(cnt) >= 5:
        # fit an ellipse to the contour
        ellipse = cv2.fitEllipse(cnt)
        # calculate aspect ratio of ellipse
        aspect_ratio = min(ellipse[1])/max(ellipse[1])
        # check if aspect ratio is close to 1 to filter for circular shapes
        if 0.2 <= aspect_ratio <= 0.7:
            area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
            hole_areas.append(area)
            hole_count += 1
            # draw ellipse on original image
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

	
# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(thresh, keypoints, blank, (0, 0, 255),
						cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Elliptical Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs

cv2.imshow("image", image)


cv2.imshow("Filtering Elliptical Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
