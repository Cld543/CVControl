import cv2
import numpy as np
import copy
import pyautogui as pag
import math

bg_captured = False # Flag to determine if background has been captured.
cap = cv2.VideoCapture(0)
cap.set(10, 200)

bg = None # The captured background image to subtract
bg_region_x = 0.5
bg_region_y = 0.7
screen_width, screen_height = pag.size()
mouse_position = (screen_height // 2, screen_width // 2)
pag.FAILSAFE = False
pag.MINIMUM_SLEEP = 0.001
click_ready = False
click_distance = 0
current_mouse = None
prev_mouse = None

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

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    
 
    
    # Draw rectangle around area that background is going to be captured from
    cv2.rectangle(frame, (int(bg_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(bg_region_y * frame.shape[0])),
                 (255, 0, 0), 2)
    
    cv2.putText(frame, "Press Space to Capture Background", (25, 25), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
    cv2.putText(frame, "From Area Within the Rectangle", (25, 50), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
    cv2.imshow('CVControl', frame)
    
    if bg_captured:
        image = subtract_bg(frame)
        image = image[0:int(bg_region_y * frame.shape[0]),
                    int(bg_region_x * frame.shape[1]):frame.shape[1]]
        
        # Apply a variety of masks to the image to make it easier to detect 
        # the contours within it.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray", gray)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        #cv2.imshow("Blue", blur)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Mask", thresh)
        
        #thresh2 = copy.deepcopy(thresh)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_length = len(contours)
        max_area = -1
        
        if contour_length > 0:
            for i in range(contour_length):
                con = contours[i]
                con_area = cv2.contourArea(con)
                if con_area > max_area:
                    max_area = con_area
                    max_index = i
            result = contours[max_index] # Get the max contour
            hull = cv2.convexHull(result)
            
            # Create a blank image to display contours
            drawing = np.zeros(image.shape, np.uint8)
            

            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
             
            cv2.resize(drawing, (screen_height, screen_width))
            hull = cv2.convexHull(result, returnPoints=False)
            defects = cv2.convexityDefects(result, hull)

            image_dims = image.shape
            
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
                
                finger_x = top_point[1]
                finger_y = top_point[0]
                
                prev_mouse = current_mouse
                current_mouse = top_point
                delta = dist(prev_mouse, current_mouse)
                
               
                mouse_position = map_mouse_position(finger_x, finger_y, im_x, im_y) 
                dist_left_to_center = dist(left_point, centroid)
                    
                if dist_left_to_center < click_distance and click_ready:
                        #pag.click(mouse_position[0], mouse_position[1])
                        print("Click!")
                        
                cv2.circle(drawing, centroid, 7, (255, 0, 0), -1)
                cv2.circle(drawing, top_point, 7, (255, 0, 255), -1)
                cv2.circle(drawing, left_point, 7, (255, 150, 0), -1)
                cv2.circle(drawing, right_point, 7, (20, 150, 255), -1)
                cv2.line(drawing, left_point, centroid, (0, 255, 255), 2)
               
                if dist_left_to_center > click_distance + 15:
                    pag.moveTo(mouse_position[0], mouse_position[1])
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
        print("---Background Reset---")
    elif k == ord('c'):
        print("---Ready to Detect Clicks---")
        click_ready = True
        click_distance = dist(left_point, centroid) / 1.2
        