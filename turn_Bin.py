import cv2
import numpy as np


def turn_Bin(img_file="./images/Metal_1.jpg"):
  
    img = cv2.imread(img_file)


    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the threshold for "black"
    black_threshold = 25

    mask = gray < black_threshold

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours as ellipses
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img, ellipse, (0, 255, 0), 1)

    # Print the number of circles drawn
    num_Circles = {len(contours)}

    return (img, num_Circles)

img, num_circles = turn_Bin()
cv2.imshow('Result', img)
print(f"Number of circles drawn: {num_circles}")
cv2.waitKey(0)
cv2.destroyAllWindows()

