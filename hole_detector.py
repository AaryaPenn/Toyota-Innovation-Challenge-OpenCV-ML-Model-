import cv2
import numpy as np
import turn_Bin

def hole_Detect():
    img, num_circles = turn_Bin()
    cv2.imshow('Result', img)
    print(f"Number of circles drawn: {num_circles}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 


hole_Detect()
