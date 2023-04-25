import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.hierarchy import fcluster
import process_data
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

def patients_corr_analysis(X):
    CorrPatients=np.corrcoef(X)
    #cluster patients according to correlations
    row_linkage = hierarchy.linkage(
    distance.pdist(CorrPatients), method='complete')
    col_linkage = hierarchy.linkage(
    distance.pdist(CorrPatients.T), method='complete')
    cl=sns.clustermap(CorrPatients, row_linkage=row_linkage,col_linkage=col_linkage,figsize=(20, 20), cmap="YlGnBu")
    return row_linkage

#What is inside the clusters ? 
def get_clusters(X,linkage,nb_clusters):
    fl = fcluster(linkage,nb_clusters,criterion='maxclust')
    clusters=[]
    for i in range(nb_clusters):
        df=X.iloc[np.where(fl==i+1)]
        clusters.append(df)
    return clusters

def get_gene_stats(cluster):
    return cluster.describe()

def get_MRD_stats(cluster,y):
    return pd.concat([cluster,y],axis=1).groupby('MRD Response').count().iloc[:,1][0]/len(cluster)*100

def patients_clusters_analysis(X,y,linkage,nb_clusters):
    clusters=get_clusters(X,linkage,nb_clusters)
    for i in range (len(clusters)):
        print('Le pourcentage de patients qui sont détectés MRD- est : ',get_MRD_stats(clusters[i],y),',le nombre de patients dans le cluster est : ',len(clusters[i]))
    pass

def main(data,nb_clusters):
    X,y=process_data.split_x_y(data)
    linkage=patients_corr_analysis(X)
    patients_clusters_analysis(X,y,linkage,nb_clusters)



