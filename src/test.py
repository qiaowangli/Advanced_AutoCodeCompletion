#!/usr/bin/python3
import storage_connection as sc
import Word2Vec as w2v
import traversal as tr
import sys

"""
Last edited by   : Shawn
Last edited time : 08/11/2021
Version Status: stable
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data = sc.storage_connection(str(sys.argv[1]))  # call storage_connection() to get the ast
    root = w2v.ast2lcrs(ast_data[0])  # call ast2lcrs() to convert the ast to lcrs

    in_order_list = tr.in_order_traversal(root)
    print(in_order_list)
    reverse_preorder = tr.reverse_preorder(root)
    print(reverse_preorder)
