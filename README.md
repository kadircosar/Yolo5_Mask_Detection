# Face Mask detection with Yolov5
   <img width="400" src=images/maksssksksss313.png></a>
   <img width="400" src=images/maksssksksss442.png></a>

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

We create a directory called data_mask to keep our dataset now. This directory needs to be in the same folder as the yolov5 repository folder we just cloned.
```bash
mkdir data_mask
```
Download the dataset.

https://www.kaggle.com/andrewmvd/face-mask-detection/download

Unzip and move it to data_mask folder that you created.

# Convert the Annotations into the YOLO v5 Format
In this part, we convert annotations into the format expected by YOLO v5. There are a variety of formats when it comes to annotations for object detection datasets.

Annotations for the dataset we downloaded follow the PASCAL VOC XML format, which is a very popular format. The PASCAL VOC format stores its annotation in XML files where various attributes are described by tags. Since this a popular format, you can find online conversion tools. 

But we created  python files that convert  PASCAL VOC XML to Yolo format.
All you need to do make sure you are in the  directory that in the same folder as the yolov5 repository folder you cloned.

```bash
wget https://github.com/kadircosar/Yolo5_Mask_Detection/archive/refs/heads/main.zip
unzip main.zip
```
You need to copy your annotations path for running python script, or just move python scripts -that you just dowlanded- to in data_mask folder.
İf you move your scripts to in data_mask folder just run with:

```bash
python3 xml_into_YOLO_txt.py --path annotations
```
Or you can also copy annotations path and runs with it.For me its home/kadir/githubprojects/Yolo5_Mask_Detection/data_mask/annotations 

For example:

```bash
python3 xml_into_YOLO_txt.py --path home/kadir/githubprojects/Yolo5_Mask_Detection/data_mask/annotations
```
# Testing the annotations
Just for a sanity check, let us now test some of these transformed annotations. We randomly load one of the annotations and plot boxes using the transformed annotations, and visually inspect it to see whether our code has worked as intended.

Run with terminal:

```bash
python3 testing_annotations.py --path annotations
```
# Partition the Dataset for train, test, validataion 
Next we partition the dataset into train, validation, and test sets containing 80%, 10%, and 10% of the data, respectively. You can change the split values according to your convenience.

Run this code in data_mask folder path terminal:
We create the folders to keep the splits.
```bash
!mkdir images/train images/val images/test annotations/train annotations/val annotations/test
```
Now train_test_split and move it folders:
```bash
python3 move_image_annotations_folders.py --path annotations
```

Rename the annotations folder to labels, as this is where YOLO v5 expects the annotations to be located in.
```bash
mv annotations labels
```
# Data config file  
Details for the dataset you want to train your model on are defined by the data config YAML file. The following parameters have to be defined in a data config file:

train, test, and val: Locations of train, test, and validation images.
nc: Number of classes in the dataset.
names: Names of the classes in the dataset. The index of the classes in this list would be used as an identifier for the class names in the code.
Create a new file called face_mask_data.yaml and place it in the yolov5/data folder. Then populate it with the following.
```bash
train: ../data_mask/images/train/ 
val:  ../data_mask/images/val/
test: ../data_mask/images/test/

# number of classes
nc: 3

# class names
names: ['with_mask', 'without_mask', 'mask_weared_incorrect']
```
YOLO v5 expects to find the training labels for the images in the folder whose name can be derived by replacing images with labels in the path to dataset images. For example, in the example above, YOLO v5 will look for train labels in
# Train
Now all you need to do make sure you activate your venv and  make sure you are in the directory  that in yolov5.


Finally, run the training:

```bash
python3 train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 100 --data face_mask_data.yaml --weights yolov5s.pt --workers 24 --name mask_det
```
This might take up to 30 minutes to train, depending on your hardware.

# Detect
There are many ways to run inference using the detect.py file.

The source flag defines the source of our detector, which can be:

A single image
A folder of images
Video
Webcam
...and various other formats. We want to run it over our test images so we set the source flag to ../data_mask/images/test/.

The weights flag defines the path of the model which we want to run our detector with.
conf flag is the thresholding objectness confidence.
name flag defines where the detections are stored. We set this flag to mask_det; therefore, the detections would be stored in runs/detect/mask_det/.
With all options decided, let us run inference over our test dataset.
```bash
python3 detect.py --source ../data_mask/images/test/ --weights runs/train/mask_det/weights/best.pt --conf 0.25 --name mask_det
```
Also you can run with your webcam realtime and detect masks
```bash
python3 detect.py --source --0 --weights runs/train/mask_det/weights/best.pt --conf 0.25 --name mask_det
```

# Test
We can use the  val.py  file to compute score of test set.To perform the evaluation on our test set, we set the task flag to test.The script calculates for us the Average Precision for each class, as well as mean Average Precision.

Run:
```bash
python3 val.py --weights runs/train/mask_det/weights/best.pt --data face_mask_data.yaml --task test --name test_yolo
```

Here is the test's ouput:

   <img  src=images/test.png></a>

Things like plots of various curves (F1, AP, Precision curves etc) can be found in the folder runs/val/test_yolo.Let's see some of these graphs.

Confusion matrix:

<img width="800" src=images/confusion_matrix.png></a>

Precision:

<img width="800" src=images/P_curve.png></a>
# Weights
If you want to detect and test whitout training, you can just copy the folder-named runs- in yolov5. Then run codes in terminal.


And that's it.You can also use diffrent datasets in this project.İf your dataset annotations in yolo format just skip "Convert the Annotations into the YOLO v5 Format" .İf it's not yolo or pascal voc format you can find online conversion tools. 

