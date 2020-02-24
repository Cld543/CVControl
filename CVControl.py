import cv2
import numpy as np
import pyautogui as pag
import imutils


cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2(history = 100, 
                                        varThreshold = 15, 
                                        detectShadows = False)
lower = np.array([0, 20, 70], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret2, thresh = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)
    cv2.imshow('Frame', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
