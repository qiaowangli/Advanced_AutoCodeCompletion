#!/usr/bin/python3
from src import storage_connection as sc
import numpy as np

"""
Last edited by   : Shawn
Last edited time : 29/11/2021
Version Status: dev
TO DO:
The functions in this file are for reading and preparing the inputs for the NN.
Required: Path to NN_input.txt
          Path to vector.csv
"""


def get_input_vectors_and_labels(credential_path):
    """
    @ input : path to NN_input.txt, path to vectors.csv
    @ output: input vectors, labels as numpy arrays
    """
    inputs = []
    labels = []

    # Get the word embedding table as a df
    # word_embedding_df = get_word_embedding_table(csv_file_path, 0.9)
    # This call can be replaced wih
    word_embedding_df = sc.storage_connection_embedding(credential_path, "pca_lookup_table.csv")
    # sequence_list = get_seq_list(seq_file_path)
    # This call can be replaced with
    sequence_list = sc.storage_connection_sequence(credential_path, "NN_input.txt")

    for seq in sequence_list:
        # Replace the current integer with its corresponding vector in the word embedding table if > 0,
        # else use vector of all 0's
        inputs.append([list(word_embedding_df.loc[val - 1]) if val > 0 else [0] * 34 for val in seq[:-1]])
        # Store the last integer in each sequence as the label
        labels.append([list(word_embedding_df.loc[val - 1]) if val > 0 else [0] * 34 for val in seq[-1:]])

    # Convert the inputs and labels to numpy arrays
    inputs = np.array(inputs, dtype=float)
    labels = np.array(labels, dtype=float)

    return inputs, labels
