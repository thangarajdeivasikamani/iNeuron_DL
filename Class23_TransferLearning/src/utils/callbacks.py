import tensorflow as tf
import os
import numpy as np
import time
import logging

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name
    
def get_callbacks(config, X_train,log_dir,CKPT_path):

     #Visuvalize the data using tensorflowboard
    tensorboard_logs_dir = config["logs"]["tensorboard_logs"]
    tensorboard_log_dir_path = os.path.join(log_dir, tensorboard_logs_dir)
    os.makedirs(tensorboard_log_dir_path, exist_ok=True)
    tensorboard_log_name = config["logs"]["tensorboard_logs_name"]
    unique_tensorboard_log_name = get_timestamp(tensorboard_log_name)
    path_to_tensorboard_log = os.path.join(tensorboard_log_dir_path, unique_tensorboard_log_name)
    logging.info(f"savings logs at: {path_to_tensorboard_log}")
    print(path_to_tensorboard_log)
    #Create the file write
    file_writer = tf.summary.create_file_writer(logdir=path_to_tensorboard_log)
    # Call the file_writer using with command like file open
    with file_writer.as_default():
    # Here we passing the image b/w 10 to 30  image & the reshape size we can dynamically adjust based on 
    # given input input 10,20 images ,so we will give -1 based on input image will adjust, last one is define for dimention(Grey scale)
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)
    #Call the Tensorboard
    tf.keras.callbacks.TensorBoard(log_dir=path_to_tensorboard_log)
     #Create the callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=path_to_tensorboard_log)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
      # Create the check point
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]