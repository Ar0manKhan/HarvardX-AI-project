import imp
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dense,
    DepthwiseConv2D,
)

from sklearn.model_selection import train_test_split

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
    model.evaluate(x_test, y_test, verbose=2)

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
    # getting count of all images to create numpy array of that size
    count = 0
    for i in range(NUM_CATEGORIES):
        count += len(os.listdir(os.path.join("gtsrb", str(i))))

    reshaped_image = np.empty((count, IMG_HEIGHT, IMG_WIDTH, 3), dtype="uint8")
    image_label = np.empty(count, dtype="uint8")
    c = 0
    dimension = (IMG_HEIGHT, IMG_WIDTH)
    for i in range(NUM_CATEGORIES):
        dir_path = os.path.join("gtsrb", str(i))
        for file in os.listdir(dir_path):
            reshaped_image[c] = cv2.resize(
                cv2.imread(os.path.join(dir_path, file)), dimension
            )
            image_label[c] = i
            c += 1
    return (reshaped_image, image_label)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 3)))
    model.add(DepthwiseConv2D((3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
