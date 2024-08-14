import cv2
from ultralytics import YOLO
import pandas as pd

# load our csv file
csv = 'Cleanroom Tracking.csv'
df = pd.read_csv(csv)

# initialize video capture
filename = '3_2024-07-18_11-18-17.mp4'
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)

#initialize start time
date = filename[:-4].split('_')[1]
starttime = list(map(int, filename[:-4].split('_')[2].split('-')))
machine = 'Brewer00'

# set our confidence threshold
conf_threshold = 0.7

# initialize our trained model for image recognition
model = YOLO("best1.pt")
white, teal, navy = 0, 0, 0

def frames_to_timecode(frame, fps, start):
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
    hr0, min0, sec0 = start
    startseconds = hr0 * 3600 + min0 * 60 + sec0 + 9
    seconds = frame // fps + startseconds
    hr, min, sec = seconds // 3600, (seconds // 60) % 60, seconds % 60
    
    return f'{int(hr):02d}:{int(min):02d}:{int(sec):02d}'
    

# confirm the video capture is initialized
if not cap.isOpened():
    print("Error: Unable to open video stream")

# pixels to crop the video to zoom in on the specific machine
topy, boty, topx, botx = 230, 460, 480, 755

length = 60 #seconds
frame = 0
num = 0

while True:
    # skip frames to process every `length` seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    ret, img = cap.read()
    if not ret:
        print("Error: Unable to read frame from video stream!")
        break

    # crop and process the image
    cropped = img[topx:botx+1, topy:boty+1]
    results = model(cropped)

    # save the cropped image for reference
    cv2.imwrite(f'video_frames/cropped{num}.jpg', img)
    

    # initialize detection counters
    detected_classes = {0: 0, 1: 0, 2: 0}
    color_ids = {0: 'navy', 1: 'teal', 2: 'white'}

    # loop over the results
    for result in results:
        for data in result.boxes.data.tolist():
            _, _, _, _, confidence, class_id = data
            class_id = int(class_id)

            if confidence >= conf_threshold:
                detected_classes[class_id] += 1

                color = color_ids[class_id]
                
                # calculate time and send data to csv
                print(f'{color} suit is using the machine at image {num}')
                time = frames_to_timecode(frame, fps, starttime)
                new_data = [date, time, color, machine]
                df.loc[len(df)] = new_data
                df.to_csv(csv, mode='w', header=True, index=False)
                break
    
    # reset counters if no one detected
    if all(value == 0 for value in detected_classes.values()):
        print(f'for figure {num}, no one was detected.')
        print(f'at image {num} the time is supposed to be {frames_to_timecode(frame, fps, starttime)}')
        navy = teal = white = 0

    num += 1
    frame += int(length * fps)