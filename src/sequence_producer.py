#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics


"""
Last edited by   : Shawn
Last edited time : 15/11/2021
Version Status: dev
TO DO: Verify correctness
"""


class new_Node:
    def __init__(self, json_object):
        self.id = json_object['id']
        self.type = json_object['type']
        self.child = None
        self.next = None
        self.parent = None
        try:
            self.value = json_object['value']
        except KeyError as err:
            self.value = None


def addSibling(parent_node, node, child_node):
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
        return addSibling(node, node.child, child)
    else:
        node.child = child
        node.child.parent = node
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
    return root, lookup_table


def sequenceSplit(in_order_list, lookup_table):
    """
    @input : an entire ast list
    @output: sub_sequence table -> Key: training_id, Value: sub_sequence
    """
    subSequence_list = {}
    training_dataset_index = 0
    list = []
    for index in in_order_list:
        if(lookup_table[index].child is None):
            list.append(lookup_table[index].value)
        else:
            list.append(lookup_table[index].type)

        if(lookup_table[index].next != None):
            # we need to get the parent nodes starting from the end of "in_order_list"
            number_of_parent = 1
            seeking_node = lookup_table[index] # we use this seeking node to find out the number of parents we have.
            while(seeking_node.parent.id != 0):
                number_of_parent += 1
                seeking_node=seeking_node.parent
            # now we extract the parents nodes 
            while(-number_of_parent < 0):
                list.append(lookup_table[in_order_list[-number_of_parent]].type)
                number_of_parent -= 1
            
            subSequence_list[training_dataset_index]=list # terminate the list
            training_dataset_index += 1 # add up the index number
            list = [] # earse the whole list
    return subSequence_list


def get_key(val, lookup_table):
    for key, value in lookup_table.items():
        if val == value:
            return key


def tokenization(sub_sequence_dict):
    """
    @input : dict containing the ID and AST branch
    @output: tokenized_dict --> input dict just replace words with int representation
                and lookup_table_dict --> {1, [R = 'Program, F = 12],...}
    """
    lookup_table_dict = {}
    lookup_table_a = {}
    lookup_table_b = {}
    tokenized_subsequence = []
    token_index = 1
    frequency = 1
    # Creating the lookup table for the provided sub_sequence_dict
    for sequence in sub_sequence_dict:
        temp_list = []
        for val in sub_sequence_dict[sequence]:
            # Check if the value is already within the dictionary
            if val not in lookup_table_a.values():
                lookup_table_a[token_index] = val
                token_index += 1

            new_val = get_key(val, lookup_table_a)
            temp_list.append(new_val)

            lookup_table_b[val] = frequency
            frequency += 1

        tokenized_subsequence.append(temp_list)

    # Now I need to make my final dict which contains the mapping of {value: [word, frequency]}
    for index in lookup_table_a:
        x = lookup_table_a[index]  # lookup table for val
        y = lookup_table_b[x]  # lookup
        lookup_table_dict[index] = [x, y]

    return tokenized_subsequence, lookup_table_dict


def csv_to_df(file):
    """
    @input : csv file
    @output: data frame containing the csv file
    """
    df = pd.read_csv(file)
    return df


def standardize_subsequence(tokenized_subsequence):
    """
    @input : tokenized sequence
    @output: standardized tokenized sequence: Sequences that are uniform in length
    """
    standardized_subsequence = []
    cutoff = 16  # this value was determined from analysis of our tokenized_subsequence

    # I need to add 0s to any sequence that is < 16 in length and cut off any sequence that is > than 16
    for sequence in tokenized_subsequence:
        sequence = sequence[::-1]  # reverse the list

        if len(sequence) == cutoff:
            standardized_subsequence.append(sequence)

        elif len(sequence) < cutoff:
            # padd 0s to make it 16 in length
            while len(sequence) != cutoff:
                sequence.append(0)
            standardized_subsequence.append(sequence)

        else:  # len(sequence) > cutoff
            # I need to ignore everything after the 16'th value
            standardized_subsequence.append(sequence[:16])

    # # code below was used for understanding what cutoff point to choose
    # dist = {}
    # for x in tokenized_subsequence:
    #     seq_len = len(x)
    #     dist[seq_len] = 0
    #
    # for x in tokenized_subsequence:
    #     seq_len = len(x)
    #     dist[seq_len] += 1
    #
    # plt.bar(dist.keys(), dist.values(), 1.0, color='g')
    # plt.show()
    #
    # values = []
    # for sequence in tokenized_subsequence:
    #     values.append(len(sequence))
    # median = statistics.median(values)
    # print(median)

    return standardized_subsequence

