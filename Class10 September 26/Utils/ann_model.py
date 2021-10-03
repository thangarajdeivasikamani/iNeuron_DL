import tensorflow as tf
import pandas as pd
import numpy as np
from Utils import ploting

def ann_model(loss_function,optimizer,metrics,epochs,X_train,y_train,validation_input,batch_size):

    # we need to flatten the input means 28x 28 convert to sigle array
      LAYERS = [tf.keras.layers.Flatten(input_shape=[28,28],name = "inputLayer"),
          tf.keras.layers.Dense(300,activation="relu",name="hiddenLayer1"),
          tf.keras.layers.Dense(100,activation="relu",name="hiddenLayer2"),
          tf.keras.layers.Dense(10,activation="softmax",name = "outputlayer")] # Flat the input as 784 input
    # we will create the model from above layer , 
    # we are creating as sequential model, becasue we are not skip any layer
      model_clf = tf.keras.models.Sequential(LAYERS)
      model_clf.layers
      print(model_clf.summary())
    # firstlayer ,hidden layer,output layer param calcualtion
    # input* Weight +bias
    # 300 input the hidden layer + 100 neuron units, so 100 weights and 100 bias
    # 100 input to output layer + 10 neuron units, so 10 weights and 10 bias
    # (784*300 + 300), (300*100 +100),(100*10 + 10)
      print(f"first layer name:{model_clf.layers[1].name}")
      weights, biases = model_clf.layers[1].get_weights()
      print(f"wight size  of the first layer:{len(weights)}")
      print(f"bias size of the first layer:{len(biases)}")
      model_clf.compile(loss=loss_function ,optimizer=optimizer, metrics=metrics)
      print(f"Epoc number:{epochs}")
      print(f"X_train_size:{len(X_train)}")
      print(f"y_train_size:{len(y_train)}")
      print(f"Number of samples per:{len(X_train)/batch_size}")
      history = model_clf.fit(X_train, y_train, epochs=epochs, validation_data=validation_input,
      batch_size=batch_size)
      print(f"Histroy details:{history.params}")
      hist_dataframe =pd.DataFrame(history.history)
      print(f"History dataframe:{hist_dataframe.head()}")
      ploting.accuracy_plt(hist_dataframe)
      return model_clf
def evaluate(modelname,X_test, y_test):
    modelname.evaluate(X_test, y_test)

def predict(modelname,X_predict):
    X_new = X_predict

    y_prob = modelname.predict(X_new)

    y_prob.round(3)
    Y_pred= np.argmax(y_prob, axis=-1)
    print(f"Predicted output:{Y_pred}")
    return Y_pred