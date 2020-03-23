import cv2
import numpy as np

bg_captured = False # Flag to determine if background has been captured.
cap = cv2.VideoCapture(0)
bg = None # The captured background image to subtract
bg_region_x = 0.3
bg_region_y = 0.9

def subtract_bg(frame):
    fgmask = bg.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    image = cv2.bitwise_and(frame, frame, mask=fgmask)
    return image

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    
    # Draw rectangle around area that background is going to be captured from
    cv2.rectangle(frame, (int(bg_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(bg_region_y * frame.shape[0])),
                 (255, 0, 0), 2)
     
    cv2.imshow('Initial Frame', frame)
    
    if bg_captured:
        image = subtract_bg(frame)
        image = image[0:int(bg_region_y * frame.shape[0]),
                    int(bg_region_x * frame.shape[1]):frame.shape[1]]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        cv2.imshow("Mask", thresh)
    # Exit if the Escape key is pressed
    k = cv2.waitKey(10)
    
    if k == 27: 
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord(' '):
        bg = cv2.createBackgroundSubtractorMOG2(0, 60)
        print("---Captured Background---")
        bg_captured = True