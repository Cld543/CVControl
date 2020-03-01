import cv2
import numpy as np
import pyautogui as pag
import imutils


cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2(history = 100, 
                                        varThreshold = 15, 
                                        detectShadows = False)
lower = np.array([0, 90, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024, 768))
    frame = cv2.flip(frame, 1)
    #fgmask = bg.apply(frame)
    #cv2.imshow("Input", fgmask)
    #cv2.imshow("Input", frame)

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    skin = cv2.erode(skin, kernel, iterations = 3)
    skin = cv2.dilate(skin, kernel, iterations = 3)
    
    skin = cv2.GaussianBlur(skin, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skin)
    
    threshold = cv2.threshold(skin, 0, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=2)
    
    cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgmask)
    
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
