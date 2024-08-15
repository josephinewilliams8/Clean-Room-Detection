import cv2
from ultralytics import YOLO
import os

# initialize variables and model
conf_threshold = 0.4
model = YOLO("yolov8n.pt")
pic = 0

def save_photos(path):
    """Function to save photos which can be later be trained to a folder
    under the name "cleanroom_pics"

    Args:
        path (str): path to folder containing .mp4 videos
    """
    global pic
    cap = cv2.VideoCapture(path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # boundary in which we want to crop each frame to
    x1, x2, y1, y2 = 230, 460, 480, 755
    
    for frame in range(nframes):
        ret, img = cap.read()
        image = img[y1:y2+1, x1:x2+1]
        if ret == False:
            break
        
        if frame%65==0:
            # detect objects in cropped frame that contains machine
            results = model(image)

            # UNCOMMENT PRINT STATEMENT BELOW: see more details about what our model detects in the frame.
            # print(results[0].boxes)

            # loop over the results
            for result in results:
                # loop over the detections
                for data in result.boxes.data.tolist():
                    _, _, _, _, confidence, class_id = data
                    class_id = int(class_id)
                    
                    # check if there is a person near our machine (person: class_id == 0)
                    # if there is a person, check the color of their suit using color masking.
                    if class_id == 0:
                        if confidence > conf_threshold:
                            cv2.imwrite(f'cleanroom_pics/photo{pic}.jpg', image)
                            print(frame)
                            pic += 1

# save images from video footage to create a training set that can be annotated
folderpath = '<PATH TO FOLDER CONTAINING SAMPLE FOOTAGE>'
for file in os.listdir(folderpath):
        if file.endswith(".mp4"):
            path = os.path.join(folderpath, file)
            save_photos(path)