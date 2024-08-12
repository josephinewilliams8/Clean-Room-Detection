import cv2
from ultralytics import YOLO

# initialize video capture
cap = cv2.VideoCapture('<VIDEO PATH HERE>')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# set our confidence threshold
conf_threshold = 0.7

# initialize our trained model for image recognition
model = YOLO("best1.pt")
white = 0
teal = 0
navy = 0

def frames_to_timecode(frame, fps):
    """Given the fps of a given video, will give
    the timestamp of any specific frame.

    Args:
        frame (int): the number frame which we would
        like to know the time stamp of
        fps (float): number of frames per second in the
        relevant video. 

    Returns:
        str: string giving the timestamp of the desired frame.
    """
    seconds = round(frame/fps)
    
    hr = round(seconds/3600)
    
    min = round(seconds/60) - hr*60
    
    sec = seconds - hr*3600 - min*60
    
    return f'the time is {hr}:{min}:{sec}'
    

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
    if frame%(length*fps) == 0:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

        # show the cropped image
        cropped = img[topx:botx+1, topy:boty+1]
        
        # detect color suit in cropped frame that contains machine
        results = model(cropped)

        # save what the cropped image looks like to judge results of model
        cv2.imwrite(f'video_frames/cropped{num}.jpg', image)
        
        # UNCOMMENT PRINT STATEMENT BELOW: see more details about what our model detects in the frame.
        # print(results[0].boxes)

        # loop over the results
        for result in results:
            count = 0
            
            # loop over the detections
            for data in result.boxes.data.tolist():
                _, _, _, _, confidence, class_id = data
                class_id = int(class_id)
                if count != 0:
                    continue
                
                if class_id == 0:
                    count += 1
                    # print(f'navy suit spotted with {100*confidence:.2f}% confidence')
                    navy += 1
                    if navy >= 2:
                        print(f'navy suit is using the machine at image {num}')
                if class_id == 1:
                    count += 1
                    # print(f'teal suit spotted with {100*confidence:.2f}% confidence')
                    teal += 1
                    if teal >= 2:
                        print(f'teal suit is using the machine at image {num}')
                if class_id == 2:
                    count += 1
                    # print(f'white suit spotted with {100*confidence:.2f}% confidence')
                    white += 1
                    if white >= 2:
                        print(f'white suit is using the machine at image {num}')
                
            if count == 0:
                print(f'for figure {num}, no one was detected.')
                navy = 0
                teal = 0
                white = 0
        num += 1
    frame += 1