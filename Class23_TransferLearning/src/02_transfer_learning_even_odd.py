#My previous layers already identified the digit.
#Now I need to use same layer & we can able to identify even or odd
#Backprogaration will not happen for Freeze the weight
#Suppouse you are getting less accuracy compare
#May be unfreeze one more layer
#Example we trianed for animal mode & we can't use for human prediction.

import argparse
import os
import pandas as pd
from src.utils.model import save_plot
from src.utils.common import read_config
from src.utils.data_mgmt import get_data,update_even_odd_labels
from src.utils.model import create_model,save_model,get_model_summary,save_summary,evaluate
import logging
import tensorflow as tf
import numpy as np
import time

def training(config_path):
  
    #Read the config file
    config = read_config(config_path)
    #Constrcut the logging directory
    log_dir = config["logs"]["logs_dir"]
    general_log_dir = config["logs"]["general_logs"]
    log_dir_path = os.path.join(log_dir, general_log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    log_name = config["logs"]["log_name"]
    path_to_log = os.path.join(log_dir_path, log_name)
    logging.basicConfig(filename=path_to_log, filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    logging.info('----------------------------------------------------------------------')
    start = time.time()  
    logging.info(f"Tensorflow Version: {tf.__version__}")
    logging.info(f"Keras Version: {tf.keras.__version__}")
    logging.info(f"Based Model Starting Time: {start}")

     # Get the dataset
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
      ## set the seeds
    seed = config["params"]["SEED"]   ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid])
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["model_name"]
     ## load the base model - 
    base_model_path = os.path.join(artifacts_dir, model_dir, model_name)
    base_model = tf.keras.models.load_model(base_model_path)
     ## log our model summary information in logs
    model_summary_string = get_model_summary(base_model)
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    summary_dir =config["artifacts"]["summary_dir"] 
    summary_dir_path = os.path.join(artifacts_dir, summary_dir)
    os.makedirs(summary_dir_path, exist_ok=True)
    summary_file_name = config["artifacts"]["summary_name"]
    save_summary(model_summary_string,summary_file_name,summary_dir_path)
    #log Summary info into logging
    logging.info(f"Base model summary: \n{get_model_summary(base_model)}")
   
    ## freeze the weights
    ## For Testing pupouse or to increase the accuracy we can unfreeze one more layers
    ## Or We can add one more hidden layer.
    ## default will tried with -1
    ## Here you can print with default 
    ## e. g for layer in base_model.layers  -- You will get all layer  info ( 5 layers)
    # Now if you give -2 last two layers are True True only.

    for layer in base_model.layers[: -1]:
        print(f"trainable status of before {layer.name}:{layer.trainable}")
        layer.trainable = False
        print(f"trainable status of after {layer.name}:{layer.trainable}")
    # Verify really it is freezed
    ## Verify the accuracy, When I freeze 2 layers , Accuracy staring from 0.73
    ## Now only freeze last and add one more layer with 200 dense starting 72 but we are getting 86.50
    ## without 3 hidden layer.
    print('----------------------------')
    for layer in base_model.layers:
         print(f"trainable status of after {layer.name}:{layer.trainable}")

    base_layer = base_model.layers[: -1]
    # ## define the model and compile it
    new_model = tf.keras.models.Sequential(base_layer)
    # tf.keras.layers.Dense(200, name="hiddenlayer3"),
    # tf.keras.layers.LeakyReLU(),
    new_model.add(
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
    )
   #log new model Summary info into logging
    logging.info(f"ODD and EVEN model summary: \n{get_model_summary(new_model)}")
    ## Train the model
    LOSS_FUNCTION = config["params"]["binary_loss_function"]
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = config["params"]["metrics"]
    new_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS) 
       ## Train the model
    history = new_model.fit(
        X_train, y_train_bin, # << y_train_bin for our usecase
        epochs=10, 
        validation_data=(X_valid, y_valid_bin), # << y_valid_bin for our usecase
        verbose=2)


    #Save model
    model_dir = config["artifacts"]["model_dir"]    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["even_odd_model_name"]
    save_model(new_model, model_name, model_dir_path)
    logging.info(history.params)
    #Create the plot
    history_dataframe = pd.DataFrame(history.history)
    # save the history into csv file:
    csv_file_name = os.path.join(log_dir_path,"even_odd_History.csv")
    #print(csv_file_name)
    history_dataframe.to_csv(csv_file_name )
    logging.info(f"base model is saved at {model_dir_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test, y_test_bin)}")  
    end = time.time()
    elapsed = end-start
    logging.info(elapsed)
    
    #Save the plot
    plot_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir,plot_dir)
    plot_name = config["artifacts"]["plot_name"]
    os.makedirs(plot_dir_path, exist_ok=True)    
    save_plot(history_dataframe, plot_name, plot_dir_path)

    logging.info(f"Based Model end Time: {end}")
    logging.info(f"Based Model execution Time: {elapsed}")
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
       