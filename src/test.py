#!/usr/bin/python3
import storage_connection as sc
import sequence_producer as seq_produce
import traversal as tr
import sys

"""
Last edited by   : Shawn
Last edited time : 14/11/2021
Version Status: dev
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str(sys.argv[1]))  # call storage_connection() to get the ast

    seq_table = {}  # key: id, value: sub_seq
    seq_table_index = 0
    for ast_id in range(len(ast_data)):
        root, lookup_table = seq_produce.ast2lcrs(ast_data[ast_id])  # call ast2lcrs() to convert the ast to lcrs
        in_order_list = tr.in_order_traversal(root)  # call in_order_traversal() to get the sequence.
        subSequence_list = seq_produce.sequenceSplit(in_order_list, lookup_table)
        for list in range(len(subSequence_list)):
            seq_table[seq_table_index] = subSequence_list[list]
            seq_table_index += 1

    """
    the following command would take 3-6 hours to produce the embedding_table for RNN/LSTM given the tokenized_subSequence and tokenized_lookup_table,
    To overcome this timing issue, we outputed the embedding_table to vector.csv that could be found in S3 storage, for further construction, checkout the RNN folder
    """
    tokenized_subSequence, tokenized_lookup_table = seq_produce.tokenization(seq_table)

    # This takes care of step 1 that we discussed
    # file = ''  # Need to add the filepath for the requested CSV file
    # a = w2v.csv_to_df(file)

    # This takes care of step 2 that we discussed
    standard_subsequence = seq_produce.standardize_subsequence(tokenized_subSequence)

    # Writing the input for our NN to a txt file so it can be used and avoid a 5min creation
    output_file = open('NN_input.txt', 'w')
    for sequence in standard_subsequence:
        output_file.write(str(sequence))
        output_file.write('\n')
    output_file.close()

