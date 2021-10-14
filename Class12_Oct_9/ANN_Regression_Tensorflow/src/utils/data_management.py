import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

def get_data(validation_datasize,test_size_value,random_state):
    dataset_df = pd.read_csv(".//dataset//kc_house_data.csv")
    logging.info(dataset_df.shape)
    #Drop the uncessary column like id,
    dataset_df = dataset_df.drop('date',axis=1)
    dataset_df = dataset_df.drop('id',axis=1)
    dataset_df = dataset_df.drop('zipcode',axis=1)
    # Split the data X and Y
    X = dataset_df.drop('price',axis =1).values
    logging.info("x_shape:",X.shape)
    y = dataset_df['price'].values
    # Trian and Test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = test_size_value, random_state = random_state)
    # scale the X_train, X_test set as well
    X_train_full,X_test = scaled_data(X_train_full,X_test)
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize] , X_train_full[validation_datasize:] 
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:] 
    logging.info(f"X_train.shape:{X_train.shape}")
    logging.info(f"y_train.shape:{y_train.shape}")
    logging.info(f"X_valid.shape:{X_valid.shape}")
    logging.info(f"y_valid.shape:{y_valid.shape}")
    logging.info(f"X_test.shape:{X_test.shape}")
    logging.info(f"y_test.shape:{y_test.shape}")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test),(X_train_full,y_train_full)


def scaled_data(X_train_full,X_test):
     sc = StandardScaler()
     scaled_X_train = sc.fit_transform(X_train_full.astype(np.float))
     scaled_X_test = sc.transform(X_test.astype(np.float))
     return(scaled_X_train,scaled_X_test)

def scale_predict_data(X_train_full,X_new):
     sc = StandardScaler()
     _ = sc.fit_transform(X_train_full.astype(np.float))
     X_new = sc.transform(X_new)
     return(X_new)
    

