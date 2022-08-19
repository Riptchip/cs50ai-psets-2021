import csv

import pickle as pk
import numpy as np
import tensorflow as tf


def main():

    f = open('splitted dataset.pickle', 'rb')
    x_train, x_test, y_train, y_test = pk.load(f)

    csvfile = open('ai types.csv', newline='')
    reader = csv.reader(csvfile)
    
    n_conv_n_poll = [int(x) for x in next(reader)[1:]]
    n_filters = [int(x) for x in next(reader)[1:]]
    kernel_sizes = [int(x) for x in next(reader)[1:]]
    pool_sizes = [int(x) for x in next(reader)[1:]]
    activations = next(reader)[1:]
    losses = next(reader)[1:]
    optmizers = next(reader)[1:]
    n_layers = [int(x) for x in next(reader)[1:]]
    layers_sizes = [int(x) for x in next(reader)[1:]]
    dropout_values = [float(x) for x in next(reader)[1:]]

    model = get_model(activations, losses, optmizers)


def get_models(activations, losses, optmizers):

    for function in activations:
        for loss in losses:
            for optmizer in optmizers:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense()
                ])
    
    return model


if __name__ == "__main__":
    main()