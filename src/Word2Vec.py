#!/usr/bin/python3
from storage_connection import storage_connection
import sys

"""
Last edited by   : Roy
Last edited time : 06/11/2021
TO DO: complete the else statement
"""

class Node_tree:
    def __init__(self , P):
        self.value = P
        self.child = []
        self.next = []
        for i in range(30): #max number of child
            self.child.append(None)
        for i in range(30): #max number of node in the same level
            self.next.append(None)

class Node_ast:
    def __init__(self , json_object):
        self.id     = json_object['id']
        self.type  = json_object['type']
        try:
            self.value   = json_object['value']
        except KeyError as err:
            self.value   = None

 
def ast2lcrs(filePath):
    """
    @ input : your credential_path
    @ output: LCRS tree
    """
    ast_data=storage_connection(filePath)
    root=None
    token_list={} # key: ast_id , value : coressponding Node_ast
    for ast_id in range(len(ast_data)): # loop through the whole dataset
        for token_id in range(len(ast_data[ast_id])): # loop through a single AST
            if root == None:
                # token_list[token_id]=Node_ast(ast_data[ast_id][token_id])
                root=Node_tree(ast_data[ast_id][token_id]['id'])
                root.child=ast_data[ast_id][token_id]['children']
            else:
                """
                #TO DO:
                append the id number in the LCRS tree.
                """
            break
        break


if __name__ == "__main__":
    ast2lcrs(str(sys.argv[1]))