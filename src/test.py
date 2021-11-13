#!/usr/bin/python3
import pandas as pd

import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
import sys
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
    # root, lookup_table = w2v.ast2lcrs(ast_data[0])  # call ast2lcrs() to convert the ast to lcrs
    # in_order_list = tr.in_order_traversal(root)  # call in_order_traversal() to get the sequence.
    # subSequence_list = w2v.sequenceSplit(in_order_list, lookup_table)

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

    values = []
    for x in tokenized_lookup_table:
        values.append(tokenized_lookup_table[x][0])

    filepath = '/Users/shawnnettleton/Documents/vectors.tsv'
    count = 0
    # Open the input_file in read mode and output_file in write mode
    with open(filepath, 'r') as read_obj, \
            open('output_vector.tsv', 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Append the default text in the row / list
            row.append(values[count])
            # Add the updated row / list to the output file
            csv_writer.writerow(row)
            count += 1

    # print(tokenized_subSequence)
    # for list in range(len(subSequence_list)):
    #     print(subSequence_list[list])
