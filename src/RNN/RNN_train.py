#!/usr/bin/python3
# from sklearn.decomposition import PCA
import RNN_input as rnn_in

"""
Last edited by   : Roy
Last edited time : 13/11/2021
Version Status: stable
TO DO: 1. look_up_table , the purpose of (file_path,row_zie,col_size) is to convert the data from vector.csv to numpy dataframe, then we need a look up table to know what each row is.
"""
def csv_convert(file_path,row_zie,col_size,sequence_list,look_up_table,pca_variance):
    """
    @ input : file_path,row_zie,col_size,sequence_list,look_up_table,pca_variance
    @ output: RNN trainning model
    """
    dataframe=rnn_in(file_path,row_zie,col_size,sequence_list,pca_variance)
    
    #now we have a dataframe where each row implies a dinstint word, the sequecne would be the input of RNN.
