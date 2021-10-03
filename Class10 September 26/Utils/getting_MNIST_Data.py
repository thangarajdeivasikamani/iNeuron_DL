import tensorflow as tf
def collect_input_dataset():

    #tf.config.list_physical_devices("GPU") # Checking how many GPU avalaible
    #tf.config.list_physical_devices("CPU") # Checking how many cpu avalaible
    mnist =tf.keras.datasets.mnist
    (X_train_full,y_train_full),(X_test,y_test)= mnist.load_data()
    print(f"X_train__full_shape:{X_train_full.shape}")
    print(f"y_train_full_shape:{y_train_full.shape}")
    print(f"X_train_first_element_shape:{X_train_full[0].shape}")
    print(f"X_test_shape:{X_test.shape}")
    print(f"y_test_shape:{y_test.shape}")
    return(X_train_full,y_train_full,X_test,y_test)


