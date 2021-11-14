#!/usr/bin/python3
import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
import sys
import pickle
from Bert.Bert import *

"""
Last edited by   : Roy
Last edited time : 13/11/2021
Version Status: stable
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str(sys.argv[1]),"top100.json")  # call storage_connection() to get the ast
    root,lookup_table = w2v.ast2lcrs(ast_data[0])  # call ast2lcrs() to convert the ast to lcrs
    in_order_list = tr.in_order_traversal(root) # call in_order_traversal() to get the sequence.
    subSequence_list=w2v.sequenceSplit(in_order_list,lookup_table)
    model = trainBertModel(subSequence_list)
