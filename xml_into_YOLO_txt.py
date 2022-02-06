import os
from xml_to_dict import extract_info_from_xml
from dict_into_yolotxt import convert_to_yolov5
from tqdm import tqdm
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to the annotations", default="annotations")

args = vars(ap.parse_args())
PATH = args["path"]


class Config:
    path: str = PATH


# Get the annotations
annotations = (glob.glob(os.path.join(Config.path, '*.xml')))
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict, Config.path)

