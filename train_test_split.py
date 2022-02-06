import os
from sklearn.model_selection import train_test_split
import glob


def train_test_split_(path):
    # Read images and annotations
    images = (glob.glob(os.path.join(path, '*.png')))
    annotations = (glob.glob(os.path.join(path, '*.txt')))
    images.sort()
    annotations.sort()

    print(len(images), len(annotations))
    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations,
                                                                                    test_size=0.2, random_state = 1)

    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5, random_state = 1)

    return train_images, train_annotations, val_images, test_images, val_annotations, test_annotations
