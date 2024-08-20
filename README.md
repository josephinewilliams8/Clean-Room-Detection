# Cleanroom Suit Detection
In this repository, we use an object recognition model to detect people wearing different coloured labcoats 
working on a Brewer Resist Machine in a cleanroom. The model uses weights that are generated using transfer 
learning on the pre-trained YOLOv8 model. 

Click on [this link](https://github.com/ultralytics/ultralytics) to learn more about YOLO models!

The weights were created by annotating 120 images of workers in the cleanroom in their different colours. 
After image augmentations were performed, over 450 new images were added to the dataset for training. 

# Running the Program
Before running the program, create two folders in the working directory. Name one of the folders 'cleanroom_pics,' and the other folder 'video_frames.' If not already made, the python file "create_csv.py" creates a template CSV file which will store the information of the date, time, company, and machine that a worker is using. 

The main code is in the file "mach_use_vid_detect.py," and the weights for the program are found in the file "best1.pt." 

To run the program, enter the path to the folder of video footage that you would like to investigate. The path can be found by right-clicking the folder in finder, and then clicking 'Copy as Path,' or 'Copy Address.'

For confidentiality reasons, the company names are replaced by the colours of their suit. This information can be edited in line 107 of "mach_use_vid_detect.py."

# Retraining Model
**SAVING IMAGES TO ANNOTATE**
To start off, we are going to save frames from security footage where a human has been detected. These frames will be used to train our model on the different coloured suits that are worn in the cleanroom. We will first need a folder that contains a small set of security footage -- it is helpful if the footage is from different days, to make sure we have a variety of data. For the first run-through of this program, three 15-minute videos were used. 

Copy the path to this folder containing the small set of security footage, then open up the script 'cleanroom_photos.py.' Insert the path in line 70, which is currently set to be:

        folderpath = r'<INSERT PATH TO SAMPLE FOOTAGE FOLDER HERE>'

Then, run the script. If done correctly, then the program will iterate through the videos in the folder, saving frames which contain a person in them to the folder 'cleanroom_pics.' The default is that the script will save uncropped images to the folder, but by commenting out the line:

        cv2.imwrite(f'cleanroom_pics/photo{pic}.jpg', raw_img)

and uncommenting the line:

        cv2.imwrite(f'cleanroom_pics/photo{pic}.jpg', machine_cropped)

as well as fixing line 30 with the correct cropped dimensions (which can be determined with find_crop_dimensions.py), the script will save cropped images to the folder. 

**SAVING DATASET OF IMAGES**
Once we have our cleanroom_pics folder, we can start creating our annotated dataset. This method uses [Roboflow](https://roboflow.com/), because of its annotation tools, augmentation tools, and ability to separate into training/testing sets in a data.yaml file. 

Once signed into Roboflow, create a project if one doesn't already exist. Set up the project for 'Object Detection' when given the option. Then, as Roboflow asks for the classes, list the color names you would like to detect and separate them with commas. 

Next, select the cleanroom_pics folder to upload images. After they are uploaded, use Roboflow's tools to manually annotate each image with any suits/suit colors that are observed. When it comes to annotating images, refer to the following examples of well labeled/poorly labeled data:

INSERT EXAMPLE HERE

Once all of the images have been annotated, check over the set to make sure everything is labeled properly and choose the partitions for the training/testing sets. A good separation might be 70% training, 15% validation, and 15% testing. 

You might want to check the health of the data to make sure that certain colours are not too underrepresented. If they are, you can go into video footage to save more frames of those specific colours and add those into the dataset by repeating prior steps. 

Now, we can generate the dataset. For pre-processing, some steps are to auto orient, resize,and add augmentation; these pre-processing steps are done to add more photos to the dataset with slightly adjusted features to improve our model's robustness. After augmentations are complete, the data is ready to be saved.  

Click 'Export Dataset' and save as a zip file. Once the dataset is downloaded (it should be a ZIP folder which ends in .yolov8), 'Extract All' from the folder and save the folder to a Google Drive account (the path of this folder should be copied in place of **INSERT FOLDER PATH HERE** in the code extract below).

Now, log into [Google Colab](https://colab.google.com/). In Google Colab, change runtime to T4 GPU, and make sure to Mount Google Drive. Copy the following code:
    
    !pip install ultralytics
    
    from ultralytics import YOLO

    model = YOLO('yolov8s.pt')

    model.train(data='/content/drive/MyDrive/<INSERT FOLDER PATH HERE>/data.yaml', epochs=80)

Run each line in order. This process can take around 30 minutes, but may be longer or shorter depending on how many epochs are used. If more epochs are used, the model may appear more accurate but increases the risk of overfitting the model to the training data. 

Once the program has completed, using the navigation folders on the left side, go into 'content>runs>detect>train>weights' and you should see the file 'best.pt.' This file contains the weights that are used in our program, so save the .pt file to your computer and update line 140 in 'mach_use_vid_detect.py.'

# Selecting Machine
In order to crop the selected frame to locate a particular machine, open the file 'find_crop_dimensions.py.' After inserting the file path to any video frame from the security footage in line 19 (.jpg or .png images work best), run the code. Click and drag your mouse to create a box around the machine. 

After lifting the mouse, there will be a print statement with the dimensions x1, x2, y1, and y2. Copy these dimensions, and update line 82 in 'mach_use_vid_detect.py' with these dimensions to track equipment usage by that machine. 
