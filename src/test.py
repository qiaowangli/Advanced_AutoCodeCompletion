#!/usr/bin/python3
import storage_connection as sc
import sequence_producer as seq_produce
import traversal as tr
import word_embedding as we
import sys
import pickle

"""
Last edited by   : Roy
Last edited time : 14/11/2021
Version Status: dev
TO DO: add the command for bert
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str(sys.argv[1]),objectName="new.json")  # call storage_connection() to get the ast

    seq_table={} # key: id, value: sub_seq
    seq_table_index=0
    for ast_id in range(len(ast_data)):
        root,lookup_table = seq_produce.ast2lcrs(ast_data[ast_id])  # call ast2lcrs() to convert the ast to lcrs
        in_order_list = tr.in_order_traversal(root) # call in_order_traversal() to get the sequence.
        subSequence_list=seq_produce.sequenceSplit(in_order_list,lookup_table)
        for list in range(len(subSequence_list)):
            seq_table[seq_table_index]=subSequence_list[list]
            seq_table_index+=1

    """
    the following command would take 3-6 hours to produce the embedding_table for RNN/LSTM given the tokenized_subSequence and tokenized_lookup_table,
    To overcome this timing issue, we outputed the embedding_table to vector.csv that could be found in S3 storage, for further construction, checkout the RNN folder
    """
    tokenized_subSequence, tokenized_lookup_table = seq_produce.tokenization(seq_table)
    # embedding_table=we.word_embedding(tokenized_subSequence, tokenized_lookup_table)

    """
    The seq_table would be the input for bert.
    """
    data = []
    for i in seq_table.values():
        if(None not in i):
            data.append(i)
        else:
            print(i)
    pickle.dump(data,open("fullList.p","wb"))
