import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import normalize

# Function to return the constant value columns of a given DataFrame
def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]


def process(csv_file):
    my_data=pd.read_csv('all_data.csv')
    my_data.index=my_data['Patient_id']
    my_data=my_data.drop(['Patient_id'],axis=1)
    drop_col = remove_constant_value_features(my_data)
    my_data=my_data.drop(drop_col, axis=1)
    my_data=my_data.dropna() 
    return my_data


def split_x_y(my_data):
    y=my_data['MRD Response']
    X=my_data.drop(['MRD Response'],axis=1)
    return X,y

def normalize_data(my_data,ax):
    X,y=split_x_y(my_data)
    X_normalized=normalize(X,'l1',axis=ax)
    res=pd.concat([pd.DataFrame(X_normalized,columns=X.columns,index=X.index),y],axis=1)
    return res




