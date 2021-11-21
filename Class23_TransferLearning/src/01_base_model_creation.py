
import argparse
import os
import pandas as pd
from src.utils.model import save_plot
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model,get_model_summary,save_summary,evaluate
import logging
import tensorflow as tf
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
    logging.basicConfig(filename=path_to_log, filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    logging.info('----------------------------------------------------------------------')
    start = time.time()  
    logging.info(f"Tensorflow Version: {tf.__version__}")
    logging.info(f"Keras Version: {tf.keras.__version__}")
    logging.info(f"Based Model Starting Time: {start}")

     # Get the dataset
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    
   
    #  Create the model
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    LEARNING_RATE = config["params"]["learning_rate"]
    METRICS = config["params"]["metrics"]
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS,LEARNING_RATE)

    ## log our model summary information in logs
    model_summary_string = get_model_summary(model)
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    summary_dir =config["artifacts"]["summary_dir"] 
    summary_dir_path = os.path.join(artifacts_dir, summary_dir)
    os.makedirs(summary_dir_path, exist_ok=True)
    summary_file_name = config["artifacts"]["summary_name"]
    save_summary(model_summary_string,summary_file_name,summary_dir_path)
    #log Summary info into logging
    logging.info(f"model summary: \n{get_model_summary(model)}")
   
    ## Train the model
    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET)

    #Save model
    model_dir = config["artifacts"]["model_dir"]    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)
    logging.info(history.params)
    #Create the plot
    history_dataframe = pd.DataFrame(history.history)
    # save the history into csv file:
    csv_file_name = os.path.join(log_dir_path,"Base_History.csv")
    #print(csv_file_name)
    history_dataframe.to_csv(csv_file_name )

    #Save the plot
    plot_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir,plot_dir)
    plot_name = config["artifacts"]["plot_name"]
    os.makedirs(plot_dir_path, exist_ok=True)    
    save_plot(history_dataframe, plot_name, plot_dir_path)
    #evaluate the model
    evaluate(model,X_test,y_test)
    end = time.time()
    elapsed = end-start
    logging.info(elapsed)
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
       