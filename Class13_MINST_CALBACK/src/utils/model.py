import tensorflow as tf
import time
import os
import io
import matplotlib.pyplot as plt
import logging
import numpy as np

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    #if we flatten 28 x 28 will get 784 input layer
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

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

def evaluate(model,X_test, y_test):
    logging.info(model.evaluate(X_test,y_test))

def predict(model,X_new):
    y_prob = model_clf.predict(X_new)
    logging.info(y_prob.round(3))
    logging.info(Y_pred= np.argmax(y_prob, axis=-1))
