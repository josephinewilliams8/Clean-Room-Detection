import cv2
from ultralytics import YOLO

# initialize variables and model
pic = 180
conf_threshold = 0.4
model = YOLO("yolov8n.pt")

topy = 230
boty = 460
topx = 480
botx = 755

for num in range(3,4):
    video_url = f'cr{num}.mp4'
    cap = cv2.VideoCapture(video_url)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for frame in range(nframes):
        ret, img = cap.read()
        image = img[topx:botx+1, topy:boty+1]
        if ret == False:
            break
        
        if frame%65==0:
            # detect objects in cropped frame that contains machine
            results = model(image)

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
                        if confidence > conf_threshold:
                            cv2.imwrite(f'crpics/photo{pic}.jpg', image)
                            print(frame)
                            pic += 1