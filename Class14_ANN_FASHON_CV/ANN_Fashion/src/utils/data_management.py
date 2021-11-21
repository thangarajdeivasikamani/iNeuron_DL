import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
import logging

def get_data(validation_datasize):
   
    # Trian and Test split
    
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    # In this way each pixel will be in the range [0, 1]. By normalizing images we make sure that our model (ANN) trains faster.
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    # scale the test set as well
    X_test = X_test / 255.
    logging.info(f"X_train.shape:{X_train.shape}")
    logging.info(f"y_train.shape:{y_train.shape}")
    logging.info(f"X_valid.shape:{X_valid.shape}")
    logging.info(f"y_valid.shape:{y_valid.shape}")
    logging.info(f"X_test.shape:{X_test.shape}")
    logging.info(f"y_test.shape:{y_test.shape}")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test) 

  

    

