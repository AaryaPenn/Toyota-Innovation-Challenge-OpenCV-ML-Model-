
import cv2
import numpy as np
import os

def circle_finder(img_path="./images/Metal_1.jpg"):
# Set the path to the Downloads folder and the name of the image file
    downloads_folder = os.path.expanduser("~/Downloads")
    img_file = "testben1.jpg"

    # Load the image
    img = cv2.imread(img_path)

    # Halve 5
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the threshold for "black"
    black_threshold = 25

    # Create a mask of everything that is below the black threshold
    mask = gray < black_threshold

    # Create a new image that is completely white
    white_img = np.full_like(img, (255, 255, 255))

    # Apply the mask to the original image, making only the black pixels stay black and replacing all other pixels with white
    result = np.where(mask[..., None], img, white_img)

    # Apply blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(result)

    # Draw detected blobs as red circles
    im_with_keypoints = cv2.drawKeypoints(result, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Sort keypoints by area (largest to smallest)
    keypoints_sorted = sorted(keypoints, key=lambda k: k.size, reverse=True)

    # Print the largest blob
    largest_blob = keypoints_sorted[0]
    print("Largest blob: x = {}, y = {}, size = {}".format(largest_blob.pt[0], largest_blob.pt[1], largest_blob.size))

    # Store the center of each generated ellipse
    centers = []

    # Draw each ellipse and store its center
    for keypoint in keypoints:
        # Get the coordinates and size of the ellipse
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        size = int(keypoint.size)
        # Draw the ellipse
        cv2.ellipse(im_with_keypoints, (x, y), (size, size), 0, 0, 360, (255, 0, 0), 2)
        # Store the center of the ellipse
        centers.append((x, y))

    # Print the center of each ellipse and its position on the image
    for i, center in enumerate(centers):
        print("Ellipse {}: center = {}, position = {}".format(i+1, center, center[::-1]))

    return im_with_keypoints, largest_blob, keypoints

im_keypoints, largest_blob, keypoints_Hole = circle_finder()
# Show the result
cv2.imshow('Result', im_keypoints)

im_stickerpoints, largest_sticker, keypoints_Stick = circle_finder("./images/Metal_2.jpg")

cv2.imshow('Results_sticker', im_stickerpoints)
print(largest_blob.size, '\n', largest_sticker.size)

cv2.waitKey(0)
cv2.destroyAllWindows()




