"""
Convolutional Neural Network to classify Amazon dataset

Created on TUE Apr 30 2021     10:00:00

@author: micheldearaujo

"""

import os
import sys

sys.path.append("./")
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import time
from datetime import timedelta

from src.utilities import (
    load_dataset,
    targ_shape,
    dataset_name,
    test_dataset_name,
    opt,
    base_dir,
    fbeta,
)

opt = RMSprop(learning_rate=0.01)


# Creating the CNN model, using VGG Blocks
def define_model(architecture, in_shape=targ_shape, out_shape=17):
    """
    Creates a Convolutional Neural Network model using VGG-like architecture.

    Parameters:
    - in_shape (tuple): Input shape of the images.
    - out_shape (int): Number of output classes.

    Returns:
    tf.keras.models.Sequential: CNN model.
    """
    modelo = Sequential()
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[0]),
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=targ_shape,
        )
    )
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[0]),
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[1]),
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[1]),
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[2]),
            kernel_size=(3, 3),
            dilation_rate=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    modelo.add(
        Conv2D(
            filters=int(targ_shape[0] / architecture[2]),
            kernel_size=(3, 3),
            dilation_rate=(3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(targ_shape[0], activation="relu", kernel_initializer="he_uniform"))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(out_shape, activation="sigmoid"))
    modelo.compile(optimizer=opt, loss="binary_crossentropy", metrics=[fbeta])
    print(modelo.summary())
    return modelo


# Plotting the training results
def plot_training_results(model_history):
    """
    Plots the training results including loss and Fbeta score.

    Parameters:
    - model_history (tf.keras.callbacks.History): Training history object.
    """
    # Plotting loss
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(model_history.history["loss"], color="blue", label="Training Loss")
    plt.plot(model_history.history["val_loss"], color="orange", label="Validation Loss")
    plt.legend()
    # Plotting Fbeta score
    plt.subplot(212)
    plt.title("Fbeta Score")
    plt.plot(model_history.history["fbeta"], color="blue", label="Training Fbeta")
    plt.plot(
        model_history.history["val_fbeta"], color="orange", label="Validation Fbeta"
    )
    plt.legend()
    # Saving the plot
    filename = sys.argv[0].split("/")[-1]
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename + f"_plot_{targ_shape[0]}_RMSprop.png"))
    plt.close()


# Executing the model
def run(architecture):
    """
    Executes the CNN model training and evaluates its performance.
    """
    # Load training dataset
    X_train, X_val, y_train, y_val = load_dataset(dataset_name)
    X_test, y_test = load_dataset(test_dataset_name, False)

    # Creating data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, horizontal_flip=True, vertical_flip=True, rotation_range=90
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    batch_size = 128
    print(f"Batch size: {batch_size}")
    print(f"Architecture: {architecture}")

    # Applying iterators
    train_iterator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_iterator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    test_iterator = val_datagen.flow(X_test, y_test, batch_size=batch_size)

    # Building the model
    model = define_model(architecture)
    model_name = f"cnn_{targ_shape[0]}_RMSprop_{architecture}_dropout.h5"

    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

    # Fitting the model
    start_time = time.monotonic()
    model_history = model.fit(
        train_iterator,
        steps_per_epoch=len(train_iterator),
        validation_data=val_iterator,
        validation_steps=len(val_iterator),
        epochs=200,
        verbose=1,
        callbacks=[early_stop],
    )

    # Evaluating the model on test data
    loss, fbeta = model.evaluate(test_iterator, steps=len(test_iterator), verbose=1)
    end_time = time.monotonic()
    elapsed_time = timedelta(seconds=end_time - start_time)

    print(f"> loss={loss:.3f}, fbeta={fbeta:.3f}")

    # Saving the model
    model.save(os.path.join(base_dir, model_name))

    # Plotting the learning curves
    plot_training_results(model_history)

    # Writing training details to a file
    with open(os.path.join(base_dir, "cnn_training.txt"), "a") as file:
        file.write("Training platform: CPU - Dropout RMSprop\n")
        file.write(f"Architecture: {architecture}")
        file.write(f"Image Size: {targ_shape[0]}\n")
        file.write(f"Training time: {elapsed_time}\n")
        file.write(f"Loss: {loss}\n")
        file.write(f"Fbeta_score: {fbeta}\n")
        file.write("----------------------------------------------------\n")

    print("Training time:")
    print(elapsed_time)

    return loss, fbeta


# Image division per layer
architectures = [(2, 2, 1)]

for architecture in architectures:
    try:
        # Running everything
        loss, fbeta = run(architecture)
    except:
        # Writing training details to a file
        with open(os.path.join(base_dir, "cnn_training.txt"), "a") as file:
            file.write("Training platform: CPU - Dropout\n")
            file.write(f"Architecture: {architecture}")
            file.write("FAILED!")
