import tensorflow as tf
import time
import os
import io
import matplotlib.pyplot as plt
import logging
import numpy as np
from sklearn.metrics import confusion_matrix

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS):

    LAYERS = [tf.keras.layers.Dense(units=12, name="inputLayer",input_dim=8),
          tf.keras.layers.Dense(8, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(8, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(units = 1, activation="sigmoid", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=[METRICS])

    return model_clf ## <<< untrained model

def get_unique_filename(filename):
    # Create the unique_filename
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(history_dataframe, plot_name, plot_dir):
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)
    history_dataframe.plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(path_to_plot)
    # always shows after save
    plt.show()
    plt.close()
    
def save_summary(model_summary_string,summary_file_name,summary_dir):
    unique_filename = get_unique_filename(summary_file_name)
    path_to_summary = os.path.join(summary_dir, unique_filename)
    mode = 'a' if os.path.exists(path_to_summary) else 'w+'
    with open(path_to_summary, mode) as f:
         f.write(model_summary_string)

