#!/usr/bin/python3

"""
Last edited by   : Roy
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
        node_list.append(root.id)  # Root of tree
        node_list = node_list + in_order_traversal(root.next)  # Right tree
    return node_list

# Code for the following two functions copied from: https://www.geeksforgeeks.org/given-a-binary-tree-print-all-root-to-leaf-paths/
# It was modified to store the paths in array instead of printing
def get_all_paths(root):
    """
    @ input: root of lcrs tree
    @ output: list of all paths from root to each leaf
    """
    # list to store current path
    path = []
    # list to store all paths
    all_paths = []
    get_paths_recursive(root, path, 0, all_paths)

    return all_paths

def get_paths_recursive(root, path, path_len, all_paths):
    """
    Helper function

    @input: root - root of lcrs tree
    @input: path - list representing the current path
    @input: path_len - length of current path
    @input: all_paths - list to store all root to leaf paths
    @output: None
    """
     
    # Base condition - if binary tree is
    # empty return
    if root is None:
        return

    # Add current root's information
    if(len(path) > path_len):
        path[path_len] = root.value
    else:
        path.append(root.value)
 
    # increment path_len by 1
    path_len = path_len + 1
 
    if root.child is None and root.next is None:
         
        # add leaf node then append a copy of the list
        all_paths.append(path.copy())
    else:
        # try for left and right subtree
        get_paths_recursive(root.child, path, path_len, all_paths)
        get_paths_recursive(root.next, path, path_len, all_paths)