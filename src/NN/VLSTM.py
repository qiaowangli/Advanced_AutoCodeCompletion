#!/usr/bin/python3
import sys
from data_prep import get_input_vectors_and_labels

"""
Last edited by   : Roy
Last edited time : 14/11/2021
Version Status: dev
TO DO: 
"""
def vlstm(credential):
    """
    @ input : lookup_table,sequence
    @ output: VLSTM accuracy
    """
    inputs, labels=get_input_vectors_and_labels(credential)
    print(len(inputs))

    return None

if __name__ == "__main__":
    vlstm(str(sys.argv[1]))




