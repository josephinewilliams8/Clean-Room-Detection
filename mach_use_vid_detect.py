import cv2
from ultralytics import YOLO

# initialize video capture
cap = cv2.VideoCapture('3_2024-07-17_09-30-33.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# set our confidence threshold
conf_threshold = 0.7

# initialize our trained model for image recognition
model = YOLO("best0.pt")

# confirm the video capture is initialized
if not cap.isOpened():
    print("Error: Unable to open video stream")

# pixels to crop the video to zoom in on the specific machine
topy = 230
boty = 460
topx = 480
botx = 755

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
        
        # detect color suit in cropped frame that contains machine
        results = model(img)

        # save what the cropped image looks like to judge results of model
        cv2.imwrite(f'video_frames/cropped{num}.jpg', img)
        
        # UNCOMMENT PRINT STATEMENT BELOW: see more details about what our model detects in the frame.
        # print(results[0].boxes)

        # loop over the results
        for result in results:
            count = 0
            
            # loop over the detections
            for data in result.boxes.data.tolist():
                _, _, _, _, confidence, class_id = data
                class_id = int(class_id)
                
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
        num += 1
        frame = 0
    frame += 1