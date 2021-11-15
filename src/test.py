#!/usr/bin/python3
import pandas as pd

import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
from csv import writer
from csv import reader

"""
Last edited by   : Shawn
Last edited time : 09/11/2021
Version Status: dev
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str('/Users/shawnnettleton/Documents/credentials_shawn.json'))  # call storage_connection() to get the ast

    seq_table = {}  # key: id, value: sub_seq
    seq_table_index = 0
    for ast_id in range(len(ast_data)):
        root, lookup_table = w2v.ast2lcrs(ast_data[ast_id])  # call ast2lcrs() to convert the ast to lcrs
        in_order_list = tr.in_order_traversal(root)  # call in_order_traversal() to get the sequence.
        subSequence_list = w2v.sequenceSplit(in_order_list, lookup_table)
        for list in range(len(subSequence_list)):
            seq_table[seq_table_index] = subSequence_list[list]
            seq_table_index += 1

    tokenized_subSequence, tokenized_lookup_table = w2v.tokenization(seq_table)

    # print(tokenized_subSequence)
    # for list in range(len(subSequence_list)):
    #     print(subSequence_list[list])

    # This takes care of step 1 that we discussed
    # file = ''  # Need to add the filepath for the requested CSV file
    # a = w2v.csv_to_df(file)

    # This takes care of step 2 that we discussed
    standard_subsequence = w2v.standardize_subsequence(tokenized_subSequence)
    print(standard_subsequence)
