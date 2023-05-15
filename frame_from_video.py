import cv2


video = cv2.VideoCapture('./videos/video_red_no_holes_covered_2.mov')

# Initialize a counter variable
frame_count = 0

# Loop through the video frames
while True:
    # Read the next frame from the video
    ret, frame = video.read()

    isOpen = video.isOpened()
    print(isOpen)
    # If there are no more frames, break out of the loop
    if not ret:
        
        break

    # Increment the counter variable
    frame_count += 1

    # Save the current frame as an image file
    filename = f'./frame_caps/red_no_holes_{frame_count}.jpg'
    cv2.imwrite(filename, frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press and check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()