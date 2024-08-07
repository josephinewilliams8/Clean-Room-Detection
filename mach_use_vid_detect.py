import cv2
import numpy as np
from ultralytics import YOLO

# initialize video capture
cap = cv2.VideoCapture('3_2024-07-17_09-30-33.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# the confidence threshold makes sure that we do not classify an object when we are not entirely confident about what it is.
conf_threshold = 0.4

# initialize the model for image recognition
model = YOLO("yolov8n.pt")

# confirm the video capture is initialized
if not cap.isOpened():
    print("Error: Unable to open video stream")

# pixels to crop the video to
topy = int(200/.875)
boty = int(400/.875)
topx = int(420/.875)
botx = int(660/.875)

length = 30 #seconds
frame = 0
num = 0
while True:
    ret, img = cap.read()
    
    if not ret:
        print("Error: Unable to read frame from video stream!")
        break
    
    # process image to check for any human activity every 30 seconds
    if frame == length*fps:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

        # show the cropped image
        cropped = image[topx:botx+1, topy:boty+1]

        # declaring bounds to find blue suit if it is in the cropped image
        blue_low = np.array([0,0,0], dtype=np.int64)
        blue_high = np.array([20, 13, 5], dtype=np.int64)
    
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
                    bluebool = 0
                    print(f'spotted person in frame {num} with {confidence*100:.2f}% confidence.')
                    print(confidence, 'confidence')
                    if confidence > conf_threshold:
                        if count != 0:
                            continue
                        
                        # creating a mask for the selected color.
                        blue_mask = cv2.inRange(cropped, blue_low, blue_high)
                        contour, _ = cv2.findContours(blue_mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
                        
                        # confirming the color of the suit. 
                        for cnt in contour:
                            contour_area = cv2.contourArea(cnt)
                            if contour_area > 1000:
                                if count != 0:
                                    continue
                                count = 1
                                print(f'for frame {num}, a blue suit was detected.')
                                cv2.imwrite(f'video_frames/cropped{num}.jpg', img)
                                bluebool = True
                        if not bluebool:
                            print(f'for frame {num}, a white suit was detected.')
                            cv2.imwrite(f'video_frames/cropped{num}.jpg', img)
                            count = 1
            
            if count ==0:
                print(f'for figure {num}, no one was detected.')
        num += 1
        frame = 0
    frame += 1