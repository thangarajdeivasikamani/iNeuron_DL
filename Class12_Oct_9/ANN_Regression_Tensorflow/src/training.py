
import argparse
import os
import pandas as pd
from src.utils.model import save_plot
from src.utils.common import read_config
from src.utils.data_management import get_data,scale_predict_data
from src.utils.model import create_model, get_unique_filename,save_model,get_model_summary,\
    save_summary,evaluate,predict,model_performance,model_score
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
    log_dir = config["logs"]["logs_dir"]
    general_log_dir = config["logs"]["general_logs"]
    log_dir_path = os.path.join(log_dir, general_log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    log_name = config["logs"]["log_name"]
    path_to_log = os.path.join(log_dir_path, log_name)
    logging.basicConfig(filename=path_to_log, filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    logging.warning('This will get logged to a file')
    test_size_value = config["params"]["test_size_value"]
    random_state = config["params"]["random_state"]
    # Get the dataset
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test),(X_train_full,y_train_full) = get_data(validation_datasize,test_size_value,random_state)
    
    #Visuvalize the data using tensorflowboard
    tensorboard_logs_dir = config["logs"]["tensorboard_logs"]
    tensorboard_log_dir_path = os.path.join(log_dir, tensorboard_logs_dir)
    os.makedirs(tensorboard_log_dir_path, exist_ok=True)
    tensorboard_log_name = config["logs"]["tensorboard_logs_name"]
    unique_tensorboard_log_name = get_unique_filename(tensorboard_log_name)
    path_to_tensorboard_log = os.path.join(tensorboard_log_dir_path, unique_tensorboard_log_name)
    logging.info(f"savings logs at: {path_to_tensorboard_log}")
    print(path_to_tensorboard_log)
    
    #Create the file write
    file_writer = tf.summary.create_file_writer(logdir=path_to_tensorboard_log)
    # Call the file_writer using with command like file open
    file_writer.set_as_default()
    #Call the Tensorboard
    tf.keras.callbacks.TensorBoard(log_dir=path_to_tensorboard_log)
    
    #  Create the model
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    #LEARNING_RATE = config["params"]["learning_rate"]
    METRICS = config["params"]["metrics"]
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS)
    model_summary_string = get_model_summary(model)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    summary_dir =config["artifacts"]["summary_dir"]
 
    summary_dir_path = os.path.join(artifacts_dir, summary_dir)
    os.makedirs(summary_dir_path, exist_ok=True)
    summary_file_name = config["artifacts"]["summary_name"]
    save_summary(model_summary_string,summary_file_name,summary_dir_path)
    #Create the callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=path_to_tensorboard_log)
    #https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1,  baseline=0.4,restore_best_weights=True)
    # Create the check point
    ckpt_dir =config["artifacts"]["checkpoint_dir"] 
    ckpt_dir_path = os.path.join(artifacts_dir, ckpt_dir)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    CKPT_file_name = config["artifacts"]["ckpt_name"]   
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_file_name, save_best_only=True)
    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)
    batch_size = config["params"]["batch_size"]
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET,batch_size=batch_size)

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
    score,acc = model_score(model,X_train, y_train)
    print("Training score:",score)
    print("Training accuracy:",acc)
    y_pred= predict(model,X_test) 
  
    performance = model_performance(y_test,y_pred)
    print("model_performance :", performance)

    #Predict with new customer value
    new_house = [[3,1,1180,5650,1,0,0,3,7,1180,0,1955,0,47.5112,-122.257,1340,5650]]
    x_new = scale_predict_data(X_train_full,new_house)
    y_pred= predict(model,x_new) 
    logging.info(f"New Predicted value is:{y_pred}")
    print("New Predicted value is",y_pred)
    # Load from Check point 
    #Here we can see the accuracy started from last stopped training accuracy(Means use last best weight)
    ckpt_model = tf.keras.models.load_model(CKPT_file_name)
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
    print(parsed_args)
    training(config_path=parsed_args.config)