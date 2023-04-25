import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold


def process(csv_file):
    sel=VarianceThreshold(0)
    my_data=pd.read_csv('all_data.csv')
    my_data.index=my_data['Patient_id']
    my_data=my_data.drop(['Patient_id'],axis=1)
    my_data=my_data.dropna() 
    my_data=pd.DataFrame(data=sel.fit_transform(my_data),index=my_data.index, columns=sel.get_feature_names_out(None))
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


def log2_transform (X):
    return np.log2(X)

