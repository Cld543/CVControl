import cv2
import numpy as np
import copy
import pyautogui as pag


bg_captured = False # Flag to determine if background has been captured.
cap = cv2.VideoCapture(0)
cap.set(10, 200)
bg = None # The captured background image to subtract
bg_region_x = 0.5
bg_region_y = 0.7
mouse_position = (screen_height // 2, screen_width // 2)
screen_width, screen_height = pag.size()

def subtract_bg(frame):
    fgmask = bg.apply(frame, learningRate=0)
    #kernel = np.ones((3, 3), np.uint8)
    #fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    image = cv2.bitwise_and(frame, frame, mask=fgmask)
    return image

def get_center(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            return cx, cy
        
# Maps the x and y coordinates of the inputs to the screen resolution
def map_mouse_position(x, y, im_x, im_y):
    height = (x / im_x) * screen_height
    width = (y / im_y) * screen_width
    
    return width, height


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
            
            c_x = centroid[1]
            c_y = centroid[0]
            im_x = image_dims[1]
            im_y = image_dims[0]
            
            mouse_position = map_mouse_position(c_x, c_y, im_x, im_y) 
            print("IMAGE: ", image_dims)
            print( "CURSOR: ", centroid)
            print("MOUSE: ", mouse_position)

            cv2.circle(drawing, centroid, 7, (255, 0, 0), -1)
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