import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

def get_data():
    dataset_df = pd.read_csv(".//dataset/Diabetes.csv")
    X = dataset_df.iloc[:,0:8].values
    y = dataset_df.iloc[:,-1].values
    logging.info(f"X shape:{X.shape}")
    logging.info(f"Y shape:{y.shape}")
    return (X,y)



