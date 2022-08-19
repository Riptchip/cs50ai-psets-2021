import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import pickle as pk

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
    # images, labels = load_data(sys.argv[1])
    images, labels = pk.load(open('dataset.pickle', 'rb'))

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
    
    for i in range(NUM_CATEGORIES):
        for j in range(len(os.listdir(os.path.join(data_dir, str(i))))):
            path = os.path.join(data_dir, str(i), f'{int(j / 30):05d}' + '_' + f'{j % 30:05d}' + '.ppm')
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(i)

    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            128, (4, 4), activation="elu", input_shape=(30, 30, 3)
        ),

        tf.keras.layers.AveragePooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="elu"
        ),

        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            128, (2, 2), activation="elu"
        ),

        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.Dense(128, activation="elu"),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4744)])
  except RuntimeError as e:
    print(e)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# try:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# except RuntimeError as e:
#     print(e)

# import tensorflow as tf
# from Keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7 #try various numbers here
# set_session(tf.Session(config=config))

import tensorflow as tf
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))