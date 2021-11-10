#!/usr/bin/python3
import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
import sys

"""
Last edited by   : Roy
Last edited time : 08/11/2021
Version Status: stable
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str('/Users/shawnnettleton/Documents/credentials_shawn.json'))  # call storage_connection() to get the ast
    root, lookup_table = w2v.ast2lcrs(ast_data[0])  # call ast2lcrs() to convert the ast to lcrs
    in_order_list = tr.in_order_traversal(root)  # call in_order_traversal() to get the sequence.
    subSequence_list = w2v.sequenceSplit(in_order_list, lookup_table)

    tokenized_subSequence, tokenized_lookup_table = w2v.tokenization(subSequence_list)
    print(tokenized_lookup_table)
    print(tokenized_subSequence)
    # for list in range(len(subSequence_list)):
    #     print(subSequence_list[list])
