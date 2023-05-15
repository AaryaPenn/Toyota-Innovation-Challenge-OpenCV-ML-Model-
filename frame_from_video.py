import cv2

spec_name = "red_no_holes_covered_2"

video = cv2.VideoCapture(f"./videos/video_{spec_name}.mov")


frame_count = 0


while True:
    
    ret, frame = video.read()

    isOpen = video.isOpened()
    print(isOpen)
   
    if not ret:
        break

    frame_count += 1


    filename = f'./frame_caps/{spec_name}_{frame_count}.jpg'
    cv2.imwrite(filename, frame)


    cv2.imshow('Frame', frame)

    # wait for a key press and check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()