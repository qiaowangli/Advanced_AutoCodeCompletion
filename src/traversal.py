#!/usr/bin/python3

"""
Last edited by   : Braiden
Last edited time : 07/11/2021
Version Status: stable
TO DO: Verify correctness
"""

def in_order_traversal(root):
    """
    @ input:   root of lcrs tree
    @ output:  integer list of id's in-order
    """
    node_list = []
    if root:
        node_list = in_order_traversal(root.child)
        node_list.append(root.value)
        node_list = node_list + in_order_traversal(root.next)
    return node_list