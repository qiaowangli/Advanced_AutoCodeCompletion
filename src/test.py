#!/usr/bin/python3
from storage_connection import storage_connection
from Word2Vec import ast2lcrs
from traversal import in_order_traversal
import sys

"""
Last edited by   : Braiden
Last edited time : 07/11/2021
Version Status: stable
TO DO: Verify correctness
"""

if __name__ == "__main__":
    ast_data=storage_connection(str(sys.argv[1])) # call storage_connection() to get the ast
    root=ast2lcrs(ast_data[0]) # call ast2lcrs() to convert the ast to lcrs
    in_order_list = in_order_traversal(root)
    print(in_order_list)