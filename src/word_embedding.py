#!/usr/bin/python3
import tensorflow as tf
from tensorflow.keras import layers
"""
Last edited by   : Roy
Last edited time : 09/11/2021
Version Status: dev
TO DO: Skipgram implementaion with tf
"""


def storage_connection(tokenized_subsequence, lookup_table_dict):
    """
    @ input : tokenized_dict --> input dict just replace words with int representation
                and lookup_table_dict --> {1, [R = 'Program, F = 12],...}
    @ output: word embedding table
    """
    window_size = 2
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        tokenized_subsequence[0],
        vocabulary_size=len(lookup_table_dict),
        window_size=window_size,
        negative_samples=0)
    print(len(positive_skip_grams))

