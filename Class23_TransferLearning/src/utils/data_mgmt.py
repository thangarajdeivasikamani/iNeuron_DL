import tensorflow as tf
import logging
import numpy as np

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize] ,X_train_full[validation_datasize:] 
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:] 
    logging.info(X_train.shape)
    logging.info(y_train.shape)
    logging.info(X_valid.shape)
    logging.info(y_valid.shape)
    logging.info(X_test.shape)
    logging.info(y_test.shape)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels


def update_greater_less_than_5(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        greater_than = label >= 5
        list_of_labels[idx] = np.where(greater_than, 1, 0)
    return list_of_labels
