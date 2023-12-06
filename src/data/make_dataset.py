import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from PIL import Image
from matplotlib.image import imread

sys.path.append("./")

#from src.utilities import *
from src.utilities import (
    create_tag_map,
    test_dir,
    train_dir,
    train_fnames,
    test_fnames,
    create_file_mapping,
    targ_shape,
    dataset_name,
    base_dir,
)

# Plotting some images
def plot_images(train_dir, train_fnames):
    """
    Plots a grid of images from the specified directory.

    Parameters:
    - train_dir (str): Directory path containing the images.
    - train_fnames (list): List of filenames for the images.

    Returns:
    None
    """
    for i in range(9):
        plt.subplot(330 + 1 + i)
        filename = os.path.join(train_dir, train_fnames[i + 1])
        image = imread(filename)
        plt.imshow(image)
    plt.show()


# Creating one-hot encoding
def one_hot_encode(tags, mapping):
    """
    Converts a list of tags into one-hot encoding.

    Parameters:
    - tags (list): List of tags to encode.
    - mapping (dict): Mapping from tag to numerical value.

    Returns:
    np.ndarray: One-hot encoded array.
    """
    encoding = np.zeros(len(mapping), dtype='uint8')
    for tag in tags:
        encoding[mapping[tag]] = 1
    return encoding

# Loading the dataset
def load_dataset(training_path, file_mapping, tag_mapping, targ_size):
    """
    Loads images and corresponding labels from the specified path.

    Parameters:
    - path (str): Directory path containing the images.
    - file_mapping (dict): Mapping from filename to a list of tags.
    - tag_mapping (dict): Mapping from tag to numerical value.
    - targ_size (tuple): Target size for the images.

    Returns:
    tuple: (X, y), where X is an array of images and y is an array of labels.
    """
    pics, targets = list(), list()

    for filename in os.listdir(training_path):
        print(f"\n- Loading image {filename} into size of {targ_size}")

        pic = load_img(os.path.join(training_path, filename), target_size=targ_size)
        pic = img_to_array(pic, dtype='uint8')

        tags = file_mapping[filename[:-4]]
        print(f"- Mapping the filename to the classes:")
        print(f"- Filename {filename[:-4]} converts to classes {tags}")

        target = one_hot_encode(tags, tag_mapping)
        print(f"- Tags {tags} converts to {target} after One Hot Encoding...")

        pics.append(pic)
        targets.append(target)

        # print(f"- Now we have two lists:")
        # print(f"The picture has the shape of: {np.array(pics).shape}")
        # print(f"Targets: {targets}")

    X = np.asarray(pics, dtype='uint8')
    y = np.asarray(targets, dtype='uint8')
    return X, y


if __name__ == "__main__":

    # Plotting images
    #plot_images(train_dir, train_fnames)

    # Defining the CSV filename
    mapping_csv_filename = 'train_classes.csv'

    # Defining the dataset name
    dataset_name = 'test_amazon_data_%s.npz' % (targ_shape[0])

    # Creating the dataframe with the image tags
    mapping_csv = pd.read_csv(os.path.join(base_dir, mapping_csv_filename))

    # Creating the dictionary with tag strings to numbers
    tag_mapping, _ = create_tag_map(mapping_csv)

    # Creating the map of filenames to tag lists
    file_mapping = create_file_mapping(mapping_csv)

    # Loading JPEG images into arrays
    X, y = load_dataset(test_dir, file_mapping, tag_mapping, targ_shape)

    # Saving the arrays
    np.savez_compressed(os.path.join(base_dir, dataset_name), X, y)
