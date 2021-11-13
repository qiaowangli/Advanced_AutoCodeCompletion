#!/usr/bin/python3
# from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import numpy as np
from csv import reader


"""
Last edited by   : Roy
Last edited time : 13/11/2021
Version Status: stable
TO DO: NONE
"""
def csv_convert(file_path,row_zie,col_size,pca_variance=0.9):
    """
    @ input : file_path,row_zie,col_size
    @ output: numpy dataframe
    """
    dataframe=np.zeros((row_zie,col_size))
    with open('/Users/royli/Desktop/vector.csv', 'r') as read_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        row_tracker=0
        for index in csv_reader:
            vec_list=index[0].split()
            for value_index in range(len(vec_list)):
                dataframe[row_tracker][value_index]=float(vec_list[value_index])
            row_tracker+=1
    
    # reduce the dimentions given the pca_variance value and return the reducted dataframe
    pca = PCA(n_components=pca_variance)
    pca.fit(dataframe)

    return pca.transform(dataframe)