#!/usr/bin/python3
from sklearn.decomposition import PCA
import pandas as pd

"""
Last edited by   : Roy
Last edited time : 14/11/2021
Version Status: dev
TO DO: NONE
"""


def csv_to_lookup_table(file_path, pca_variance=0.9):
    """
    @ input : file_path,pca_variance
    @ output: lookup table
    """
    dataframe = pd.read_csv(file_path)

    # reduce the dimentions given the pca_variance value and return the reducted dataframe
    pca = PCA(n_components=pca_variance)
    pca.fit(dataframe)

    # the function returns the lookup table
    return pca.transform(dataframe)


def csv_to_standard_seq(file_path):
    """
    @ input : file_path
    @ output: a set of subsequences 
    """

    return None
