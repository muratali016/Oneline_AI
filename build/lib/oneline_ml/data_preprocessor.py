import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def process(data,data_train,data_test):
    """
    This function returns X_train,X_test,y_train,y_test
    data_train: the part of the data that is going to get trained
    data_test: the part of the data that is going to predict
    """
    df_tmp=data.copy()
    print("Turning data into numbers") 
    for label,content in df_tmp.items():
        if pd.api.types.is_string_dtype(content):
            df_tmp[label]=content.astype("category").cat.as_ordered()

    df_tmp.state.cat.codes 

    print("Filling numerical missing values")
    for label,content in df_tmp.items():
        if pd.api.types.is_numeric_dtype(content):
            if df_tmp[label].isnull().sum():
                # Add a binary column which tells us if teh data was missing or not
                df_tmp[label+"_is_missing"]=pd.isnull(content)
                # Fill missing numeric values with median values
                df_tmp[label]=content.fillna(content.median())
                # Median is more robust than mean

    print("Filling and turning categorical variables into numbers")
    for label,content in df_tmp.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add a binary column to indicate whetever sample had missing value
            df_tmp[label+"_is_missing"]=pd.isnull(content)
            # Turn categories into numbers and add +1
            df_tmp[label]=pd.Categorical(content).codes+1# If there are missing valıues, pandas add automatically -1
    X_train,X_test,y_train,y_test=train_test_split(data_train,data_test,test_size=0.2)

    return X_train,X_test,y_train,y_test