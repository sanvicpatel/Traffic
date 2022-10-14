import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # Create two empty lists for images and labels
    images = []
    labels = []

    # Get every folder in data_dir
    list_of_dir = os.listdir(data_dir)

    # traverse through every folder
    for folder in list_of_dir:

        # Create path for the current folder
        path = os.path.join(data_dir, folder)

        # Checks if the path leads to a folder
        if os.path.isdir(path):

            # Get every image in current folder
            list_of_img = os.listdir(path)

            # Traverse through every image
            for image in list_of_img:

                # Create path for current image
                img_path = os.path.join(path, image)
                # Read image
                img = cv2.imread(img_path)
                # Resize image to 30, 30 dimensions
                resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                # Add image to images list
                images.append(resized)
                # Add the number of the current folder to the labels list
                labels.append(int(folder))

    # Return both lists
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create Sequential Model
    model = tf.keras.models.Sequential([

        # Convolution Layer 1
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Max-Polling Layer 1
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolution Layer 2
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Flatten Layers
        tf.keras.layers.Flatten(),

        # Hidden Layer with 256 units and dropout rate of 0.5
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output Layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
