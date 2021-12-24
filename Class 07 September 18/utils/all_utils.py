"""
  Authour : Thangaraj.D
  Email : xxx@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os
import logging

plt.style.use("fivethirtyeight") # THIS IS STYLE OF GRAPHS

def prepare_data(df):
  """It is used to separate the dependent variable and independent variables

  Args:
      df (pd.DataFrame): input the csv file

  Returns:
      tuple: it will return tuple of dependent and independent variable
  """
  logging.info('separate the dependent variable and independent variables')
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

def save_model(model, filename):
  """
  :param df: it a dataframe
  :param filename:it's path to save the plot
  :param model: trained model

  """
  logging.info('Save the model')
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model, filePath)

def load_model(predict_input,model_filename):
    logging.info('load the model')
    loaded_model = joblib.load("models/"+model_filename)
    loaded_model.predict(predict_input)

def save_plot(df, file_name, model):
      #Function inside the function-it is local function- we can use internally(kind of protected)
    def _create_base_plot(df):  # Responsible for printing the data points
        logging.info('create the plot')
        df.plot(kind="scatter",x="x1",y="x2",c="y",s=100,cmap="winter")  # x1 & x2 point, color,size,color map
        plt.axhline(y = 0, color = "black",linestyle="--",linewidth =1)   #  Yis zero, X can be anything
        plt.axvline(x = 0, color = "black",linestyle="--",linewidth =1)   #
        figure = plt.gcf()  #get current figure
        figure.set_size_inches(10,8)
    def _plot_decision_regions(X, y, classfier, resolution=0.02):
        # Plot the decision boundary
        logging.info('Plot the decision boundary')
        colors = ("red","blue","lightgreen","grey","cyan")
        cmap = ListedColormap(colors[:len(np.unique(y))])
        X = X.values # as  a array
        x1 = X[:,0] # Separate the values,all rows zero column
        x2 = X[:,1]
        x1_min,x1_max = x1.min() -1, x1.max() + 1 # Find the maximum , For increase the visible we will add +1, -1
        x2_min,x2_max = x2.min() -1, x2.max() + 1
        # Take the min & max and find the each point co-ordinate
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                  np.arange(x2_min, x2_max, resolution))
        print(xx1)
        print(xx1.ravel()) # Make it as single array.
        Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap) # Find the value xx1 & color map based on Z value
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()

    X,y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    plot_dir = "plots" # Dir name
    os.makedirs(plot_dir,exist_ok=True)#ONLY Create the model direct not avaliable
    plotPath= os.path.join(plot_dir,file_name)# will join model+ file name , Based on operating system it will automatically will join
    joblib.dump(model,plotPath)
    plt.savefig(plotPath)
    logging.info(f"save the plot at path{plotPath}")