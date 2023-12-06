"""
Models Utilities

Created on TUE Apr 30 2021     10:00:00

@author: micheldearaujo

"""
import tensorflow as tf
import numpy as np
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
import time
import datetime
from datetime import timedelta
import joblib
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Define directory paths
base_dir = "/mnt/c/Documents and Settings/miche/Documents/projects/datasets/amazonia/"
train_dir = os.path.join(base_dir, 'train-jpg')
test_dir = os.path.join(base_dir, 'test-jpg')
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# Define Model parameters
targ_shape = (64, 64, 3)  #  Image size
dataset_name = 'amazon_data_%s.npz'%(targ_shape[0]) # Dataset accordingly to the image size
test_dataset_name = 'test_amazon_data_%s.npz'%(targ_shape[0]) # Dataset accordingly to the image size
opt = SGD(learning_rate=0.01, momentum=0.9) # Model Optimizer


def load_dataset(dataset_name, training=True):
    """
    Loads the dataset for deep learning algorithms.

    Parameters:
    - dataset_name (str): Name of the dataset file.
    - training (bool): If True, splits the dataset into training and validation sets.

    Returns:
    If training=True:
        tuple: (Xtr, Xval, ytr, yval), where Xtr and Xval are training and validation sets of images,
               and ytr and yval are corresponding labels.
    If training=False:
        tuple: (X, y), where X is the set of images and y is the corresponding labels.
    """
    # Loading
    data = np.load(base_dir + '/' + dataset_name)
    X, y = data['arr_0'], data['arr_1']
    print(f"Shape of the original dataset: {X.shape}")

    if training:
        # Separating training and validation sets
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, random_state=1)

        print('\nDimensions of the vectors are:')
        print('X_train: ', Xtr.shape)
        print('y_train: ', ytr.shape)
        print('X_val: ', Xval.shape)
        print('y_val: ', yval.shape)

        return Xtr, Xval, ytr, yval
    else:
        return X, y



# Creating a function to calculate the fbeta score
def fbeta(y_true, y_pred, beta=2):
    """
    Calculates the Fbeta score.

    Parameters:
    - y_true (tf.Tensor): True labels.
    - y_pred (tf.Tensor): Predicted labels.
    - beta (float): Beta value for Fbeta score calculation.

    Returns:
    tf.Tensor: Fbeta score.
    """
    # Clipping the prediction
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true*y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred-y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true-y_pred, 0, 1)), axis=1)
    # Calculating precision
    p = tp/(tp+fp+backend.epsilon())
    # Calculating recall
    r = tp/(tp+fn+backend.epsilon())
    # Calculating Fbeta, taking the mean for each class
    bb = beta**2
    fbeta_score = backend.mean((1+bb)*(p*r)/(bb*p+r+backend.epsilon()))
    return fbeta_score


def evaluation(model, x, true):
    """
    Evaluates the model using F1 score.

    Parameters:
    - model (tf.keras.models.Model): The trained model.
    - x (np.ndarray): Input data.
    - true (np.ndarray): True labels.

    Returns:
    float: F1 score.
    """
    y_pred = model.predict(x)
    return f1_score(true, y_pred, average='samples')


# Creating a mapping
def create_tag_map(mapping_csv):
    """
    Creates a mapping of tags to numerical values.

    Parameters:
    - mapping_csv (pd.DataFrame): DataFrame containing 'tags' column.

    Returns:
    tuple: (labels_map, inv_labels_map) where labels_map is a mapping from tag to number,
           and inv_labels_map is the inverse mapping.
    """
    labels = set()
    for i in range(len(mapping_csv)):
        tags = mapping_csv['tags'][i].split(' ')
        labels.update(tags)

    labels = list(labels)
    labels.sort()
    labels_map = {labels[k]: k for k in range(len(labels))}
    inv_labels_map = {k: labels[k] for k in range(len(labels))}
    return labels_map, inv_labels_map

# Creating a file mapping
def create_file_mapping(mapping_csv):
    """
    Creates a mapping of filenames to corresponding tags.

    Parameters:
    - mapping_csv (pd.DataFrame): DataFrame containing 'image_name' and 'tags' columns.

    Returns:
    dict: A mapping from filename to a list of tags.
    """
    mapping = dict()
    for j in range(len(mapping_csv)):
        name, tags = mapping_csv['image_name'][j], mapping_csv['tags'][j]
        mapping[name] = tags.split(' ')
    return mapping


def use_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def do_not_use_gpu():

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")