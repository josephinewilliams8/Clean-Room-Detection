import cv2
from ultralytics import YOLO
import pandas as pd
import os

# https://github.com/josephinewilliams8 

# BEFORE RUNNING THIS CODE: 
# 1) make sure that any required paths are updated
# 2) if not already present, create a folder called 'cleanroom_pics'
# 3) if not already present, create a folder called 'video_frames'
# 4) refer to README.md for any other questions

def main():
    # load our csv file
    csv = 'Cleanroom Tracking.csv'
    df = pd.read_csv(csv)

    # initialize video capture
    folder_path = r'<ENTER FOLDER PATH WITH SECURITY FOOTAGE>'
    filenum = 0
    
    # pass all of the video footage in from a given folder
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            path = os.path.join(folder_path, file)
            filenum = process_cleanroom_vid(file, path, csv, df, filenum)

def process_cleanroom_vid(filename, filepath, csv, df, filenum):
    """Video footage is passed in along with the path of the file, path to the 
    CSV file, path to the dataframe, and number of the frame of the last
    object detection.
    Loads cleanroom data into the CSV file.

    Args:
        filename (str): Name of video file ending in '.mp4'
        filepath (str): Path to the file on the computer.
        csv (str): Path to CSV file that stores cleanroom data.
        df (str): Path to the dataframe which helps to update the CSV file.
        filenum (int): Count of the last frame that detected an object

    Returns:
        int: The number of the last frame that detected an object.
    """
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    #initialize start time
    date = filename[:-4].split('_')[1]
    starttime = list(map(int, filename[:-4].split('_')[2].split('-')))

    # set our confidence threshold
    conf_threshold = 0.7

    def frames_to_timecode(frame, fps, start):
        """Given the fps of a given video, will give
        the timestamp of any specific frame.

        Args:
            frame (int): the number frame which we would
            like to know the time stamp of
            fps (float): number of frames per second in the
            relevant video. 
            start (list): the start time of the video file

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
    x1, x2, y1, y2 = 230, 460, 480, 755
    machine = 'Brewer00 Resist'
    length = 60 #seconds
    frame = 0

    while True:
        # skip frames to process every `length` seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        ret, img = cap.read()
        if not ret:
            print("Error: Unable to read frame from video stream!")
            break

        # crop and process the image
        cropped = img[y1:y2+1, x1:x2+1]
        results = model(cropped)

        # save the cropped image for reference
        # cv2.imwrite(f'video_frames/cropped{num}.jpg', img)
    
        # initialize detection counters
        # note: can change color_ids to reflect different company names
        # note: if more classes are added, update the two following dictionaries
        detected_classes = {0: 0, 1: 0, 2: 0}
        color_ids = {0: 'Blue', 1: 'Green', 2: 'White'}

        # loop over the results
        for result in results:
            for data in result.boxes.data.tolist():
                _, _, _, _, confidence, class_id = data
                class_id = int(class_id)

                if confidence >= conf_threshold:
                    detected_classes[class_id] += 1
                    color = color_ids[class_id]
                    
                    # calculate time and send data to csv
                    print(f'{color} suit is using the machine in image {filenum}.')
                    cv2.imwrite(f'video_frames/cropped{filenum}.jpg', img)
                    
                    time = frames_to_timecode(frame, fps, starttime)
                    new_data = [date, time, color, machine]
                    df.loc[len(df)] = new_data
                    df.to_csv(csv, mode='w', header=True, index=False)
                    filenum += 1
                    break
        
        # reset counters if no one detected
        if all(value == 0 for value in detected_classes.values()):
            print(f'No one was detected.')

        frame += int(length * fps)
    return filenum

if __name__ == "__main__":
    
    # initialize our trained model for image recognition
    model = YOLO("best1.pt")
    
    main()
    
    
    