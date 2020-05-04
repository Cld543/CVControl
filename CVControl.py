# CVControl - Computer Vision Based Mouse Control
# 
# @Author: Chris Dellinger
# @Version: 5/4/2020
#
# This application aims to provide an alternative method for controlling a 
# computer's mouse input by using computer vision and image processing
# techniques.
#
#  
#
import cv2
import numpy as np
import pyautogui as pag
import math
import copy
from enum import Enum

# Enum used to determine current program state. Helpful when performing
# actions while interacting with the mouse through the webcam.
class State(Enum):
    START = 0,
    MOVING = 1,
    LEFT_CLICK = 2,
    RIGHT_CLICK = 3,
    MOUSE_DOWN = 4,
    MOUSE_UP = 5,
    SCROLL = 6
    
# Assign default system webcam to cap variable.
cap = cv2.VideoCapture(0)
# Increase the brightness property of the webcam to help with masking and 
# background subtraction.
cap.set(10, 200)


# Flag to determine if background has been captured.
bg_captured = False 
bg = None 
state = State.START

# Percentage of screen region to use when capturing the background image.
bg_region_x = 0.5
bg_region_y = 0.7
screen_width, screen_height = pag.size()
mouse_position = (screen_height // 2, screen_width // 2)

click_ready = False
current_fingers = 0
prev_fingers = 0
prev_state = -1
pag.FAILSAFE = False
pag.MINIMUM_SLEEP = 0.01

# Subtract the background image created using the 
# cv2::createBackgroundSubtractorMog2 function and return the foreground mask.
def subtract_bg(frame):
    fgmask = bg.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    image = cv2.bitwise_and(frame, frame, mask=fgmask)
    return image

# Uses the moments of the image to compute the center of the contour
def get_center(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            return cx, cy
        
# Maps the x and y coordinates of the inputs to the screen resolution. This is
# used to translate the coordinates of the hand mask image to the user's screen
def map_mouse_position(x, y, im_x, im_y):
    height = (x / im_x) * screen_height
    width = (y / im_y) * screen_width
    
    return width, height

# Compute the euclidean distance between points, represented as tuples.
def dist(pt1, pt2):
    if pt1 is not None and pt2 is not None:
        return math.sqrt(((pt1[0] - pt2[0]) ** 2) + (pt1[1] - pt2[1]) ** 2)

# Uses the convexity defects of the contours parameter (the points within a 
# convex hull that are farthest away from the hull that form a 'cavity' 
# within the hull) in order to count the number of fingers that are currently
# being held up in the image. Returns the number of fingers and draws a circle
# on each of the convexity defect points to indicate this.
def get_fingers(image, contours):
     hull = cv2.convexHull(contours, returnPoints = False)

     if len(hull) > 3:
        defects = cv2.convexityDefects(contours, hull)
        if defects is not None:
            count = 0
            
            for i in range(defects.shape[0]):
                # Gets the key points of the defects
                d_start, d_end, d_far, d_depth = defects[i, 0]
                start_point = tuple(contours[d_start][0])
                end_point = tuple(contours[d_end][0])
                far_point = tuple(contours[d_far][0])
                
                # Create triangle with points to calculate angle and
                # only draw points with snaller angles, so number of counts
                # will represent number of fingers.
                a = dist(start_point, end_point)
                b = dist(far_point, start_point)
                c = dist(end_point, far_point)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                
                # If angle less than 100 degrees (in radians), add one to finger
                # count and draw a circle at the convexity defect location
                if angle <= 1.74533:  
                    count += 1
                    cv2.circle(image, far_point, 7, (200, 200, 50), -1)
            return count
        
# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    
    # Draw rectangle around area that background is going to be captured from
    cv2.rectangle(frame, (int(bg_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(bg_region_y * frame.shape[0])),
                 (255, 0, 0), 2)
    
    if not bg_captured:
        cv2.putText(frame, "Press Space to Capture Background", (25, 25), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
        cv2.putText(frame, "From Area Within the Rectangle", (25, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
    else:
        cv2.putText(frame, "Press 'R' to re-capture background", (25, 25), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
        if not click_ready:
            cv2.putText(frame, "Press 'C' to start mouse control", (25, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
        
    cv2.imshow('CVControl', frame)
    
    if bg_captured:
        image = subtract_bg(frame)
        
        # Grab the screen area represented by the blue box.
        image = image[0:int(bg_region_y * frame.shape[0]),
                    int(bg_region_x * frame.shape[1]):frame.shape[1]]
        image_dims = image.shape
        # Apply a variety of masks to the image to make it easier to detect 
        # the contours within it.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray", gray)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        #cv2.imshow("Blur", blur)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Mask", thresh)
        
        thresh2 = copy.deepcopy(thresh)
        contours, hier = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_length = len(contours)
        max_area = -1
        
        if contour_length > 0:
            for i in range(contour_length):
                con = contours[i]
                con_area = cv2.contourArea(con)
                
                if con_area > max_area:
                    max_area = con_area
                    max_index = i
            
            # Get the contour with the maximum area
            result = contours[max_index] 
            hull = cv2.convexHull(result)
            
            # Create a blank image to display contours
            drawing = np.zeros(image.shape, np.uint8)
            
            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                                       
            
            
            # Extract the key points from the contour
            left_point = tuple(result[result[:,:,0].argmin()][0])
            right_point = tuple(result[result[:,:,0].argmax()][0])
            top_point = tuple(result[result[:,:,1].argmin()][0])
            bottom_point = tuple(result[result[:,:,1].argmax()][0])
            
            centroid = get_center(result)
            
            if centroid is not None:
                c_x = centroid[1]
                c_y = centroid[0]
                im_x = image_dims[1]
                im_y = image_dims[0]
                
                # Use the top-most point to control the mouse location
                finger_x = top_point[1]
                finger_y = top_point[0]
                        
                num_fingers = get_fingers(drawing, result)
                
                if num_fingers is not None:
                    num_fingers += 1
                else:
                    num_fingers = 0
            
                dist_top_to_center = dist(top_point, centroid) 

                if dist_top_to_center < 120 and num_fingers == 1:
                    num_fingers = 0
                prev_fingers = current_fingers
                current_fingers = num_fingers
                
                # Display number of fingers currently being held up.
                cv2.putText(drawing, str(num_fingers) , (25,
                           drawing.shape[0] - 25),
                           cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
               
                # Map contour coordinates to screen coordinates
                mouse_position = map_mouse_position(finger_x, finger_y, im_x, im_y) 

                # Set the program state according to the change in the number
                # of fingers held up since the previous frame
                prev_state = state
                             
                if state == State.MOVING:
                    pag.moveTo(mouse_position[0], mouse_position[1])
                    
                if prev_fingers == 0 and current_fingers == 0:
                    state = State.SCROLL                                
                    
                if prev_fingers == 2 and current_fingers == 1:
                    state = State.LEFT_CLICK
                
                if prev_fingers == 3 and current_fingers == 2:
                    state = State.RIGHT_CLICK
                    
                if prev_fingers == 5 and current_fingers == 4:
                    state = State.MOUSE_DOWN
                
                if prev_fingers == 4 and current_fingers != 4:
                    state = State.MOUSE_UP
                
                # Perform actions based on the current state of the program
                if click_ready:
                    if state == State.LEFT_CLICK:
                        pag.click(mouse_position[0], mouse_position[1])
                        print("Click!")
                        state = State.MOVING
                        
                    if state == State.RIGHT_CLICK:
                        pag.rightClick(mouse_position[0], mouse_position[1])
                        print("Right Click!")
                        state = State.MOVING
                    
                    if state == State.MOUSE_DOWN:
                        pag.mouseDown()
                        pag.moveTo(mouse_position[0], mouse_position[1])
                    
                    if state == State.MOUSE_UP:
                        pag.mouseUp()  
                        state = State.MOVING
                    
                    if state == State.SCROLL:
                        if mouse_position[1] < screen_height / 2 + 100:
                            pag.scroll(15)
                            print("Scrolling up")                           
                        elif mouse_position[1] > screen_height / 2 + 300:
                            pag.scroll(-15)
                            print("Scrolling down")
                            
                        if prev_fingers == 0 and current_fingers != 0:
                            state = State.MOVING
                            
                
                # Draw circles over all key points of hull
                cv2.circle(drawing, centroid, 7, (255, 0, 0), -1)
                cv2.circle(drawing, top_point, 7, (255, 0, 255), -1)
                cv2.circle(drawing, left_point, 7, (255, 150, 0), -1)
                cv2.circle(drawing, right_point, 7, (20, 150, 255), -1)

            cv2.resize(drawing, (screen_height, screen_width))
            # Display the contour image.
            cv2.imshow("Contours", drawing)
                
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
    elif k == ord('r'):
        bg = None
        bg_captured = False
        click_ready = False
        state = State.START
        print("---Background Reset---")
    elif k == ord('c'):
        if not click_ready:
            print("---Ready to Detect Clicks---")
            click_ready = True
            state = State.MOVING
        else:
            print("---Click Detection Disabled---")
            click_ready = False
            state = State.START
        