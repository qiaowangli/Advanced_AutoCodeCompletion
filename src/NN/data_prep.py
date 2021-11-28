#!/usr/bin/python3

"""
Last edited by   : Braiden
Last edited time : 27/11/2021
Version Status: dev
TO DO: 

The functions in this file are for reading and preparing the inputs for the LSTM. 

Required: Path to NN_input.txt
          Path to vector.csv (Nov 14, 2021 version from S3)
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def lookup_table_to_df(csv_file_path):
    """
    @ input : path to vector.csv
    @ output: dataframe containing contents of vector.csv
    """

    column_names = [ 'c' + str(i) for i in range(1,101)]
    df = pd.DataFrame(columns=column_names)
    csv_file = open(csv_file_path, 'r')

    current_row = 0
    while True:

        line = [float(el.strip('"')) for el in csv_file.readline().strip().split()]
        if(not line):
            break
        df.loc[current_row] = line
        current_row += 1
    csv_file.close()

    return df

def apply_pca_to_def(df, variance):
    """
    @ input : df = dataframe representing vector.csv, variance = amount of variance to capture
    @ output: new dataframe after applying pca
    """
    pca = PCA(n_components=variance)
    pca.fit(df)
    transformed_data = pca.transform(df)

    reduced_column_names = ['c' + str(i) for i in range(1,transformed_data.shape[1] + 1)]
    transformed_df = pd.DataFrame(transformed_data, columns=reduced_column_names)

    return transformed_df

def get_word_embedding_table(csv_file_path, variance):
    """
    @ input : path to vector.csv, variance = amount of variance to capture using pca
    @ output: word embedding table as df
    """

    df = lookup_table_to_df(csv_file_path)
    
    return apply_pca_to_def(df, variance)

def get_seq_list(seq_file_path):
    """
    @ input : path to NN_input.txt
    @ output: list of sequences(lists)
    """
    sequence_file = open(seq_file_path, 'r')
    sequence_list = []

    while True:
        line = sequence_file.readline()
        if(not line):
            break
        current_sequence = [int(x) for x in line.strip().strip('][').split(',')]
        sequence_list.append(current_sequence)
    
    sequence_file.close()
    
    return sequence_list



def get_input_vectors_and_labels(seq_file_path, csv_file_path):
    """
    @ input : path to NN_input.txt, path to vectors.csv
    @ output: input vectors, labels as numpy arrays
    """
    inputs = []
    labels = []

    # Get the word embedding table as a df
    word_embedding_df = get_word_embedding_table(csv_file_path, 0.9)
    sequence_list = get_seq_list(seq_file_path)

    for seq in sequence_list:
        # Replace the current integer with its corresponding vector in the word embedding table if > 0, else use vector of all 0's
        inputs.append([ list(word_embedding_df.loc[val -1]) if val > 0 else [0]*34 for val in seq[:-1]])
        # Store the last integer in each sequence as the label
        labels.append(seq[-1])
    
    # Convert the inputs and labels to numpy arrays
    inputs = np.array(inputs, dtype=float)
    labels = np.array(labels, dtype=float)

    return (inputs, labels)
