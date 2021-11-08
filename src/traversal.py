#!/usr/bin/python3

"""
Last edited by   : Shawn
Last edited time : 08/11/2021
Version Status: stable
TO DO: Verify correctness
"""


def reverse_preorder(root):
    """
    @ input:   root of lcrs tree
    @ output:  integer list of id's reverse preorder
    """
    node_list = []
    temp_stack = [root]
    while len(temp_stack) != 0:
        curr = temp_stack.pop()
        node_list.append(curr.value)
        if curr.child is not None:
            temp_stack.append(curr.child)
        if curr.next is not None:
            temp_stack.append(curr.next)
    return node_list


def in_order_traversal(root):
    """
    @ input:   root of lcrs tree
    @ output:  integer list of id's in-order
    """
    node_list = []
    if root:
        node_list = in_order_traversal(root.child)  # Left tree
        node_list.append(root.value)  # Root of tree
        node_list = node_list + in_order_traversal(root.next)  # Right tree
    return node_list
