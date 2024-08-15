# Cleanroom Suit Detection
In this repository, we use an object recognition model to detect people wearing different coloured labcoats 
working on a Brewer Resist Machine in a cleanroom. The model uses weights that are generated using transfer 
learning on the pre-trained YOLOv8 small model. 

The weights were created by annotating 120 images of workers in the cleanroom in their different colours. 
After image augmentations were performed, over 450 new images were added to the dataset for training. 

# Running the Program
The main code is in the file "mach_use_vid_detect.py," and the weights for the program are found in the file "best1.pt." 
If not already made, the python file "create_csv.py" creates a template CSV file which will store the information
of the date, time, company, and machine that a worker is using. 

To run the program, enter the path to the folder of video footage that you would like to investigate. The path can be found
by right-clicking the folder in finder, and then clicking 'Copy as Path.'

For confidentiality reasons, the company names are replaced by the colours of their suit. This information can be edited
in line 98 of "mach_use_vid_detect.py."

# Retraining Model
If new images need to be added to the dataset, the file "cleanroom_photos.py" saves images with a person in frame to 
a folder called 'crpics.' These images can be annotated in [Roboflow](https://roboflow.com/), which will separate the model into training, testing,
and validation sets which can be downloaded to any device. 

Once the dataset is downloaded (it should be a ZIP folder which ends in .yolov8), 'Extract All' from the folder and save 
the folder to a Google Drive account. Then, open up Google Colab.

On Google Colab, change runtime to T4 GPU, and make sure to Mount Google Drive. Copy the following code, entering one line at a time:
    
    !pip install ultralytics
    
    from ultralytics import YOLO

    model = YOLO('yolov8s.pt')

    model.train(data='/content/drive/MyDrive/<FOLDER NAME>/data.yaml', epochs=80)

Run each line in order. This process can take around 30 minutes, but may be longer or shorter depending on how many epochs 
are used. If more epochs are used, the model may appear more accurate but increases the risk of overfitting the model to 
the training data. 

Once the program has completed, using the navigation folders on the left side, go into 'content>runs>detect>train>weights' and you 
should see the file 'best.pt.' This file contains the weights that are used in our program, so save the .pt file to your
computer and update line 131 in 'mach_use_vid_detect.py.'
