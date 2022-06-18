import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def process(data):
    """
    Process:Turning data into numbers , Filling and turning categorical variables into numbers
    This function returns data that is clean but includes Nan values
    You have to use "preprocessor.fill_na()" function
    """
    df_tmp=data.copy()
    print("Turning all strings to category values") 
    for label,content in df_tmp.items():
        if pd.api.types.is_string_dtype(content):
            df_tmp[label]=content.astype("category").cat.as_ordered()
    
    LE = LabelEncoder()
    print("Filling and turning categorical variables into numbers")
    for label,content in df_tmp.items():
        if not pd.api.types.is_numeric_dtype(content):            
            df_tmp[label+"_is_missing"]=pd.isnull(content)
            df_tmp[label]=pd.Categorical(content).codes+1
            df_tmp[label] = LE.fit_transform(df_tmp[label])
    
    

    return df_tmp

def fill_na(df_tmp):
    """"
    Remove the missing values from the data
    """
    for label,content in df_tmp.items():
        if pd.api.types.is_numeric_dtype(content):
            if df_tmp[label].isnull().sum():
                
                df_tmp[label+"_is_missing"]=pd.isnull(content)
                df_tmp[label]=content.fillna(content.median())
                
    return df_tmp

def separete_sets(df_train,df_test):
    """
    This function returns X_train,X_test,y_train,y_test
    data_train: the part of the data that is going to get trained
    data_test: the part of the data that is going to predict
    """
       
    X_train,X_test,y_train,y_test=train_test_split(df_train,df_test,test_size=0.2)
      
    return X_train,X_test,y_train,y_test