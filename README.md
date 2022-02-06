# Face Mask detection with Yolov5
YOLO refers to “You Only Look Once” is one of the most versatile and famous object detection models. For every real-time object detection work, YOLO is the first choice by Data Scientist and Machine learning engineers.

İn this project we will train the YOLO v5 detector on a face mask dataset. 

# Start with cloning Yolov5
We begin by cloning the YOLO v5 repository and setting up the dependencies required to run YOLO v5. You might need sudo rights to install some of the packages.

In a terminal, type:
```bash
git clone https://github.com/ultralytics/yolov5
```
# İnstall requirements
I recommend you create a new conda or a virtualenv environment to run your YOLO v5 experiments as to not mess up dependencies of any existing project.
Once you have activated the new environment, install the dependencies using pip.

Before running this code in terminal make sure activate your venv that you created for this project and run this code in path that you cloned yolov5. 
```bash
pip install -r yolov5/requirements.txt
```
# Dowland the face mask data
With this dataset, it is possible to create a model to detect people wearing masks, not wearing them, or wearing masks improperly.
This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.
The classes are:

With mask;
Without mask;
Mask worn incorrectly.

We create a directory called face_mask_dataset to keep our dataset now. This directory needs to be in the same folder as the yolov5 repository folder we just cloned.
We create a directory called face_mask_dataset to keep our dataset now. This directory needs to be in the same folder as the yolov5 repository folder we just cloned.
```bash
mkdir face_mask_dataset
```
Download the dataset.

https://www.kaggle.com/andrewmvd/face-mask-detection/download

Unzip and move it to face_mask_dataset folder that you created.

# Convert the Annotations into the YOLO v5 Format
In this part, we convert annotations into the format expected by YOLO v5. There are a variety of formats when it comes to annotations for object detection datasets.

Annotations for the dataset we downloaded follow the PASCAL VOC XML format, which is a very popular format. The PASCAL VOC format stores its annotation in XML files where various attributes are described by tags. Since this a popular format, you can find online conversion tools. 
