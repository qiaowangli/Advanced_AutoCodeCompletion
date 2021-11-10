#!/usr/bin/python3
import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
import sys

"""
Last edited by   : Shawn
Last edited time : 09/11/2021
Version Status: dev
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str(sys.argv[1]))  # call storage_connection() to get the ast
    seq_table={} # key: id, value: sub_seq
    seq_table_index=0
    for ast_id in range(len(ast_data)):
        root,lookup_table = w2v.ast2lcrs(ast_data[ast_id])  # call ast2lcrs() to convert the ast to lcrs
        in_order_list = tr.in_order_traversal(root) # call in_order_traversal() to get the sequence.
        subSequence_list=w2v.sequenceSplit(in_order_list,lookup_table)
        for list in range(len(subSequence_list)):
            seq_table[seq_table_index]=subSequence_list[list]
            seq_table_index+=1
    print(seq_table)
