
import argparse
import os
import pandas as pd
from src.utils.log_mgmt import logging_dir
from src.utils.callbacks import get_callbacks
from src.utils.model import save_plot
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, get_unique_filename,save_model,get_model_summary,save_summary,evaluate
import logging
import tensorflow as tf
import numpy as np

def training(config_path):
    print('----------------------------------------------------------------------')
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    #Read the config file
    config = read_config(config_path)
    #Constrcut the logging directory
    log_dir=logging_dir(config)
    # Get the dataset
    CKPT_path = "model_ckpt.h5"
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    CALLBACKS_LIST=get_callbacks(config, X_train,log_dir,CKPT_path)
    #  Create the model
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    model_summary_string = get_model_summary(model)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    summary_dir =config["artifacts"]["summary_dir"]
 
    summary_dir_path = os.path.join(artifacts_dir, summary_dir)
    os.makedirs(summary_dir_path, exist_ok=True)
    summary_file_name = config["artifacts"]["summary_name"]
    save_summary(model_summary_string,summary_file_name,summary_dir_path)
   
  

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET,callbacks=CALLBACKS_LIST)

    #Save model
    model_dir = config["artifacts"]["model_dir"]    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)
    logging.info(history.params)
    #Create the plot
    history_dataframe = pd.DataFrame(history.history)
    #Save the plot
    plot_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir,plot_dir)
    plot_name = config["artifacts"]["plot_name"]
    os.makedirs(plot_dir_path, exist_ok=True)    
    save_plot(history_dataframe, plot_name, plot_dir_path)
    #evaluate the model
    evaluate(model,X_test,y_test)
    # Load from Check point 
    #Here we can see the accuracy started from last stopped training accuracy(Means use last best weight)
    ckpt_model = tf.keras.models.load_model(CKPT_path)
    history = ckpt_model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET, callbacks=CALLBACKS_LIST)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # we can add help also
    args.add_argument("--config", "-c", default="config.yaml")
    #args.add_argument("--secret", "-s", default="secret.yaml")
    #while run time we can give as python src/training.py  --config=config.yaml 
    #we can call aslo python src/training.py  -c=config.yaml  -s secret.yaml for experiment
    # even we can pass from another folder
    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)