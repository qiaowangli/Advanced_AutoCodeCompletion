#!/usr/bin/python3
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd



"""
Last edited by   : Roy
Last edited time : 13/11/2021
Version Status: dev
TO DO: NONE
"""
def csv_convert(file_path,pca_variance=0.9):
    """
    @ input : file_path,pca_variance
    @ output: reducted numpy dataframe
    """
    dataframe=pd.read_csv(file_path)
    
    # reduce the dimentions given the pca_variance value and return the reducted dataframe
    pca = PCA(n_components=pca_variance)
    pca.fit(dataframe)

    return pca.transform(dataframe)