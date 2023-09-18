import cv2
import numpy as np

holes_images = cv2.imread("./images/blobs.jpg")
gray = cv2.cvtColor(holes_images, cv2.COLOR_BGR2GRAY)
img =cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# params = cv2.SimpleBlobDetector_Params() 
# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200


holes = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2,
                            param1=50,param2=30,minRadius=0,maxRadius=20)
holes = np.uint16(np.around(holes))

for i in holes[0, :]:
    cv2.circle(holes_images, (i[0], i[1]), i[2], (0,0,255), 2)

cv2.imshow("Circle Detection", holes_images)
cv2.waitKey(0)
cv2.destroyAllWindows