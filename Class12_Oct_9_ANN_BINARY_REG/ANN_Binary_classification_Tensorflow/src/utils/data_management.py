import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

def get_data(validation_datasize,test_size_value):
    dataset_df = pd.read_csv(".//dataset//Churn_Modelling.csv")
    X = dataset_df.iloc[:,3:13].values
    y = dataset_df.iloc[:,-1].values
    #Encoding the categorical data ( convert 1,2,3)
    X[:, 1]  = data_encoding(X[:,1])
    X[:, 2]  = data_encoding(X[:,2])
    #Now DL algorithm will confuse  with priority so we need to do onehot encoding.
    # or we can do direct dummy method also.
    X = column_transfering(X) 
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = test_size_value, random_state = 0)
    # scale the X_train, X_test set as well
    X_train_full,X_test = scaled_data(X_train_full,X_test)
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize] , X_train_full[validation_datasize:] 
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:] 
    logging.info(X_train.shape)
    logging.info(y_train.shape)
    logging.info(X_valid.shape)
    logging.info(y_valid.shape)
    logging.info(X_test.shape)
    logging.info(y_test.shape)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test),(X_train_full,y_train_full)

def data_encoding(dataset_values):
    label_encoder_X = LabelEncoder()
    encoded_data = label_encoder_X.fit_transform(dataset_values)
    return encoded_data

def column_transfering(X_values):
    ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
    return(ct.fit_transform(X_values))

def scaled_data(X_train_full,X_test):
     sc = StandardScaler()
     scaled_X_train = sc.fit_transform(X_train_full)
     scaled_X_test = sc.transform(X_test)
     return(scaled_X_train,scaled_X_test)

def scale_predict_data(X_train_full,X_new):
     sc = StandardScaler()
     scaled_X_train = sc.fit_transform(X_train_full)
     X_new = sc.transform(X_new)
     return(X_new)
    

