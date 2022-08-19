import cv2
import numpy as np
import os
import sys
import tensorflow as tf

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

    images = list()
    labels = list()
    
    # Iterate through all images in data drectory
    for i in range(NUM_CATEGORIES):
        for j in range(len(os.listdir(os.path.join(data_dir, str(i))))):
            # Get the relative path of the image idependent of the plataform
            path = os.path.join(data_dir, str(i), f'{int(j / 30):05d}' + '_' + f'{j % 30:05d}' + '.ppm')

            # Resize image so that all images are the same size
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            # Add image and its respective label to their respective lists
            images.append(img)
            labels.append(i)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 128 filters using a 4x4 kernel
        tf.keras.layers.Conv2D(
            128, (4, 4), activation="elu", input_shape=(30, 30, 3)
        ),

        # Average-pooling layer, using 3x3 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3)),

        # Convolutional layer. Learn 128 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="elu"
        ),

        # Average-pooling layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 128 filters using a 2x2 kernel
        tf.keras.layers.Conv2D(
            128, (2, 2), activation="elu"
        ),

        # Average-pooling layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add 3 hidden layers with dropout relative to sizes
        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, activation="elu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.Dropout(0.25),

        # Add an output layer with output units for all types of signs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile neural network
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()