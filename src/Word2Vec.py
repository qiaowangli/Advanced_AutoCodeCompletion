#!/usr/bin/python3

"""
Last edited by   : Roy
Last edited time : 08/11/2021
Version Status: dev
TO DO: Verify correctness
"""


class new_Node:
    def __init__(self, json_object):
        self.id    = json_object['id']
        self.type = json_object['type']
        self.child = None
        self.next = None
        self.parent=None
        try:
            self.value   = json_object['value']
        except KeyError as err:
            self.value   = None

def addSibling(parent_node,node,child_node):
    if node is None:
        return None

    while node.next is not None:
        node = node.next

    node.next = child_node
    node.next.parent=parent_node
    return node.next


def addChild(node, child):
    if node is None:
        return None

    if node.child is not None:
        return addSibling(node,node.child,child)
    else:
        node.child = child
        node.child.parent=node
        return node.child


def find_node(candidateNode, targetNode):
    if candidateNode is None:
        return None

    if candidateNode.id == targetNode:
        return candidateNode
    # find left side
    left_node = find_node(candidateNode.child, targetNode)
    if left_node:
        return left_node
    # find right side
    return find_node(candidateNode.next, targetNode)


def ast2lcrs(ast):
    """
    @ input : AST tree
    @ output: LCRS tree, lookup_table
    """
    root = None
    lookup_table = {}  # key: ast_id , value : coressponding Node_object
    for token_id in range(len(ast)):  # loop through a single AST
        if root is None:
            root = new_Node(ast[token_id])
            lookup_table[token_id]=root
            for child in ast[token_id]['children']:
                lookup_table[child] = addChild(root, new_Node(ast[child]))
        else:
            # find out the location of node
            starting_node = find_node(root, token_id)  # raise error if we cannot find the node. After we find it, we would append the child under this node
            if starting_node is not None:
                try:
                    for child in ast[token_id]['children']:
                        lookup_table[child] = addChild(starting_node, new_Node(ast[child]))
                except KeyError as err:
                    pass
    return root,lookup_table

def sequenceSplit(in_order_list,lookup_table):
    """
    @input : an entire ast list
    @output: sub_sequence table -> Key: training_id, Value: sub_sequence
    """
    subSequence_list={} 
    trainning_dataset_index=0
    list=[]
    for index in in_order_list:
        if(lookup_table[index].value is None):
            list.append(lookup_table[index].type)
        else:
            list.append(lookup_table[index].value)

        if(lookup_table[index].next != None):
            # we need to get the parent nodes starting from the end of "in_order_list"
            number_of_parent=1
            seeking_node=lookup_table[index] # we use this seeking node to find out the number of parents we have.
            while(seeking_node.parent.id != 0):
                number_of_parent+=1
                seeking_node=seeking_node.parent
            # now we extract the parents nodes 
            while(-number_of_parent<0):
                list.append(lookup_table[in_order_list[-number_of_parent]].type)
                number_of_parent-=1
            
            subSequence_list[trainning_dataset_index]=list # terminate the list 
            trainning_dataset_index+=1 # add up the index number
            list=[] # earse the whole list
    return subSequence_list
