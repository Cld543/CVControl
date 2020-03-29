import cv2
import numpy as np
import copy
import pyautogui as pag
import math

bg_captured = False # Flag to determine if background has been captured.
cap = cv2.VideoCapture(0)
cap.set(10, 200)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
bg = None # The captured background image to subtract
bg_region_x = 0.5
bg_region_y = 0.7
screen_width, screen_height = pag.size()
mouse_position = (screen_height // 2, screen_width // 2)
pag.FAILSAFE = False
click_ready = False

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


def dist(pt1, pt2):
    return math.sqrt(((pt1[0] - pt2[0]) ** 2) + (pt1[1] - pt2[1]) ** 2)

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
        #cv2.imshow("Gray", gray)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        #cv2.imshow("Blue", blur)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
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
            result = contours[max_index] # Get the max contour
            hull = cv2.convexHull(result)
            
            # Create a blank image to display contours
            drawing = np.zeros(image.shape, np.uint8)
            
# =============================================================================
#             for c in contours:
#                 hull = cv2.convexHull(c)
#                 cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
#                 cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
# =============================================================================
            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
             
            cv2.resize(drawing, (1920, 1080))
            centroid = get_center(result)
            
            image_dims = image.shape
            
            left_point = tuple(result[result[:,:,0].argmin()][0])
            right_point = tuple(result[result[:,:,0].argmax()][0])
            top_point = tuple(result[result[:,:,1].argmin()][0])
            bottom_point = tuple(result[result[:,:,1].argmax()][0])
            
          
            #print(dist_left_to_center)
            
            c_x = centroid[1]
            c_y = centroid[0]
            im_x = image_dims[1]
            im_y = image_dims[0]
            
            finger_x = top_point[1]
            finger_y = top_point[0]
            
            mouse_position = map_mouse_position(finger_x, finger_y, im_x, im_y) 
            dist_left_to_center = dist(left_point, centroid)
                
            if dist_left_to_center < 70 and click_ready:
                    #pag.click(mouse_position[0], mouse_position[1])
                    print("Click!")
                    
            cv2.circle(drawing, centroid, 7, (255, 0, 0), -1)
            cv2.circle(drawing, top_point, 7, (255, 0, 255), -1)
            cv2.circle(drawing, left_point, 7, (255, 0, 255), -1)
            
            pag.moveTo(int(mouse_position[0]), int(mouse_position[1]))
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
        