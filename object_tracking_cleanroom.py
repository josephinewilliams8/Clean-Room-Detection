import numpy as np
import datetime
import cv2
from ultralytics import YOLO

# from helper import create_video_writer

# the confidence threshold makes sure that we do not classify an object when we are not entirely confident about what it is.
conf_threshold = 0.5

# initialize the model for image recognition
model = YOLO("yolov8n.pt")

# cropping image just to the box around the machine we want to detect
# update these values to change the corners of our rectangle frame
topy = 200
boty = 400
topx = 420
botx = 660

# load the images we want to investigate
for num in range(0,8):
    img = cv2.imread(f"cleanroom{num}.jpg", cv2.IMREAD_UNCHANGED)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    
    # show the cropped image
    cropped = frame[topx:botx+1, topy:boty+1]
    cv2.imwrite('cropped.jpg', cropped)

    # declaring bounds to find blue suit if it is in the cropped image
    blue_low = np.array([0,0,0], dtype=np.int64)
    blue_high = np.array([20, 13, 5], dtype=np.int64)
    blue = [blue_low, blue_high, 'blue'] 
    
    white_low = np.array([90,95,0], dtype=np.int64)
    white_high = np.array([160,160,15], dtype=np.int64)
    white = [white_low, white_high, 'white']

    # detect objects in cropped frame that contains machine
    results = model(cropped)
    mask = None

    # UNCOMMENT PRINT STATEMENT BELOW: see more details about what our model detects in the frame.
    # print(results[0].boxes)

    # loop over the results
    for result in results:
        # initialize the list of bounding boxes, confidences, and class IDs
        confidences = []
        class_ids = []
        count = 0
        
        # loop over the detections
        for data in result.boxes.data.tolist():
            _, _, _, _, confidence, class_id = data
            class_id = int(class_id)
            
            # check if there is a person near our machine (person: class_id == 0)
            # if there is a person, check the color of their suit using color masking.
            if class_id == 0:
                print(num, 'spotted person')
                print(confidence, 'confidence')
                if confidence > conf_threshold:
                    for color in [white, blue]:
                        if count != 0:
                            continue
                        
                        # creating a mask for the selected color.
                        mask = cv2.inRange(cropped, color[0], color[1])
                        colorname = color[2]
                        contour, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
                        
                        # confirming the color of the suit. 
                        for cnt in contour:
                            contour_area = cv2.contourArea(cnt)
                            if contour_area > 500:
                                if count != 0:
                                    continue
                                count = 1
                                print(f'for figure {num}, {colorname} suit was detected.')
                                cv2.imshow(f'{colorname} mask', cropped)
                                cv2.waitKey(0)
        

        if count ==0:
            print(f'for figure {num}, no one was detected.')