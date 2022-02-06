import shutil
from train_test_split import train_test_split_
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to the annotations", default="annotations")

args = vars(ap.parse_args())
PATH = args["path"]


class Config:
    path: str = PATH


train_img, train_annotations, val_img, test_img, val_annotations, test_annotations = train_test_split_(Config.path)


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# getting image path from annotation path
a = f"{Config.path}"

if len(a.split("/")) == 1:
    image_path = "images"
else:
    image_path = a.split("/")[:-1]
    image_path.append("images")
    image_path = "/".join(image_path)
    print(image_path)

# Move the splits into their folders
move_files_to_folder(train_img, f'{image_path}/train/')
move_files_to_folder(val_img, f'{image_path}/val/')
move_files_to_folder(test_img, f'{image_path}/test/')
move_files_to_folder(train_annotations, f'{Config.path}/train/')
move_files_to_folder(val_annotations, f'{Config.path}/val/')
move_files_to_folder(test_annotations, f'{Config.path}/test/')