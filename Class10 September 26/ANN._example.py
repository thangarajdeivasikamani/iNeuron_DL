import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
print(tf.__version__)
print(tf.keras.__version__)
print(pd.__version__)
from Utils import getting_MNIST_Data
from Utils import ploting
from Utils import ann_model

def main(loss_function,optimizer,metrics,epochs,batch_size):
    X_train_full,y_train_full,X_test,y_test = getting_MNIST_Data.collect_input_dataset()
    ploting.plot_single_digit_element(X_train_full[0])
    #Split training set into X_valid and X-train
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.
    VALIDATION = (X_valid, y_valid)
    modelname = ann_model.ann_model(loss_function,optimizer,metrics,epochs,X_train,y_train,VALIDATION,batch_size)
    ann_model.evaluate(modelname,X_test, y_test)
    predicted_value=ann_model.predict(modelname,X_test[:3])
    ploting.plot_predicted(X_test[:3],predicted_value, y_test[:3])


if __name__ == '__main__':
       # Before training compile the model
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"  # optimiser used to converge 
    METRICS = ["accuracy"]
    EPOCHS = 30
    BATCH_SIZE = 16
    main(loss_function=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS,epochs=EPOCHS,batch_size=BATCH_SIZE)