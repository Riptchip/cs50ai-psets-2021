import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import pickle

from sklearn.model_selection import train_test_split

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: python test.py data_directory filename.pickle')
    
    if sys.argv[1] not in os.listdir() or sys.argv[2][-7:] != '.pickle':
        sys.exit('Usage: python test.py data_directory filename.pickle')

    images, labels = load_data(sys.argv[1])

    with open(sys.argv[2], 'wb') as f:
        pickle.dump([images, labels], f)

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    with open('splitted ' + sys.argv[2], 'wb') as g:
        pickle.dump([x_train, x_test, y_train, y_test], g)
    

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


if __name__ == "__main__":
    main()