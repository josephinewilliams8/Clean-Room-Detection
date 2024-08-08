import numpy as np
import cv2
from ultralytics import YOLO

# from helper import create_video_writer

# the confidence threshold makes sure that we do not classify an object when we are not entirely confident about what it is.
conf_threshold = 0.4

# initialize the model for image recognition
model = YOLO("best.pt")

# cropping image just to the box around the machine we want to detect
# update these values to change the corners of our rectangle frame
topy = 200
boty = 400
topx = 420
botx = 660

# load the images we want to investigate
for num in range(0,3):
    img = cv2.imread(f"tester/test{num}.jpg", cv2.IMREAD_UNCHANGED)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    
    # show the cropped image
    cropped = frame[topx:botx+1, topy:boty+1]
    cv2.imwrite('cropped.jpg', cropped)

    # detect objects in cropped frame that contains machine
    results = model(img)

    # UNCOMMENT PRINT STATEMENT BELOW: see more details about what our model detects in the frame.
    # print(results[0].boxes)

    # loop over the results
    for result in results:
        count = 0
        
        # loop over the detections
        for data in result.boxes.data.tolist():
            _, _, _, _, confidence, class_id = data
            class_id = int(class_id)
            
            # check if there is a person near our machine (person: class_id == 0)
            # if there is a person, check the color of their suit using color masking.
            if class_id == 0:
                count += 1
                print(f'navy suit spotted with {100*confidence:.2f}% confidence')
            if class_id == 1:
                count += 1
                print(f'teal suit spotted with {100*confidence:.2f}% confidence')
            if class_id == 2:
                count += 1
                print(f'white suit spotted with {100*confidence:.2f}% confidence')
                    
        if count ==0:
            print(f'for figure {num}, no one was detected.')