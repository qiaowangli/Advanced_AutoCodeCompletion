#!/usr/bin/python3
from storage_connection import storage_connection
import sys

"""
Last edited by   : Roy
Last edited time : 06/11/2021
Version Status: stable
TO DO: None
"""

class new_Node:
    def __init__(self , value):
        self.value = value
        self.child = None
        self.next = None

"""
this object could be used in the further step, for this step, we use the unique id to present it.
"""
# class Node_ast:
#     def __init__(self , json_object):
#         self.id    = json_object['id']
#         self.type  = json_object['type']
#         try:
#             self.value   = json_object['value']
#         except KeyError as err:
#             self.value   = None


def addSibling(node, value):
    if (node == None):
        return None

    while (node.next!=None):
        node = node.next
    node.next = new_Node(value)
    return node.next
 
def addChild(node, value):
    if (node == None):
        return None

    if (node.child!=None):
        return addSibling(node.child, value)
    else:
        node.child = new_Node(value)
        return node.child
 
def find_node(candidateNode,targetNode):
    if candidateNode == None:
        return None

    if (candidateNode.value == targetNode):
        return candidateNode
    #find left side
    left_node = find_node(candidateNode.child, targetNode)
    if left_node:
        return left_node
    #find right side
    return find_node(candidateNode.next, targetNode)
 
def ast2lcrs(ast):
    """
    @ input : AST tree
    @ output: LCRS tree
    """
    root=None
    token_list={} # key: ast_id , value : coressponding Node_ast
    for token_id in range(len(ast)): # loop through a single AST
        if root == None:
            root=new_Node(ast[token_id]['id'])
            for child in ast[token_id]['children']:
                token_list[child]=addChild(root,child)
        else:
            # find out the location of node
            starting_node=find_node(root,token_id) # raise error if we cannot find the node. After we find it, we would append the child under this node
            if(starting_node!=None):
                try:
                    for child in ast[token_id]['children']:
                        token_list[child]=addChild(starting_node,child)
                except KeyError as err:
                    pass
    return root
    