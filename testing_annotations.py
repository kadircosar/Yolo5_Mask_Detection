import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageDraw
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to the annotations", default="annotations")


args = vars(ap.parse_args())
PATH = args["path"]


class Config:
    path: str = PATH


# Get the annotations
annotations = (glob.glob(os.path.join(Config.path, '*.txt')))


# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"with_mask": 0,
                            "without_mask": 1,
                            "mask_weared_incorrect": 2}

random.seed()

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()


# Get any random annotation file
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# getting image path from annotation path
a = f"{Config.path}"

if len(a.split("/")) == 1:
    image_path = "images"
else:
    image_path = a.split("/")[:-1]
    image_path.append("images")
    image_path = "/".join(image_path)
    print(image_path)


# Get the corresponding image file
image_file = annotation_file.replace(Config.path, image_path).replace("txt", "png")
assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)
