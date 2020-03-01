import cv2
import numpy as np
import pyautogui as pag



cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2(history = 100, 
                                        varThreshold = 15, 
                                        detectShadows = False)



if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024, 768))
    frame = cv2.flip(frame, 1)
    
  
       
    cv2.imshow('Frame', frame )
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
