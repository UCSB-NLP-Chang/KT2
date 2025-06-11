import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KC_tree.node import ParametersNode

def prune_graph(graph, prune_size=20):
    '''
    Prune the graph to ensure each node has at most one parent
    '''
    print('Before pruning: ', len(graph))
    for node in graph:
        if len(graph[node].parents) > 1:
            # Keep the first parent and remove this node from other parent's children
            for parent in graph[node].parents:
                if parent != graph[node].parents[0]:
                    parent.children.remove(graph[node])
            graph[node].parents = [graph[node].parents[0]]

    graph = prune_graph_with_question_count(graph)
    graph, mapping = merge_nodes(graph, prune_size)

    parameter_graph = dict()
    for name in graph:
        if name not in parameter_graph:
            parameter_graph[name] = ParametersNode(name)
    return graph, parameter_graph, mapping


def prune_graph_with_question_count(graph):
    '''
    Remove all nodes with no children and no question count, loop until no more nodes are removed
    '''
    has_node_removed = True
    remove_nodes = []
    while has_node_removed:
        has_node_removed = False
        for node in graph:
            if node not in remove_nodes:
                if len(graph[node].children) == 0 and graph[node].question_count == 0:
                    for parent in graph[node].parents:
                        parent.children.remove(graph[node])
                    remove_nodes.append(node)
                    has_node_removed = True
    for node in remove_nodes:
        graph.pop(node)
    # print(len(remove_nodes))
    # print(remove_nodes)
    print('After pruning: ', len(graph))
    return graph

def merge_nodes(graph, prune_size=20):
    '''
    Merge leaf nodes with the same parents that have less than prune_size questions
    '''
    mapping = dict() #Track the mapping of the merged nodes
    
    # Start from the leaf nodes and process upward
    current_nodes = [node for node in graph if len(graph[node].children) == 0]

    while len(current_nodes) > 0:
        parents = []
    
        for node in current_nodes:
            if node in mapping: # Already merged
                continue
            if len(graph[node].children) > 0: # Has children, cannot be merged
                continue
            if len(graph[node].parents) >= 1: # Not a root node
                parent = graph[node].parents[0]

                children = parent.children

                nodes_to_merge = []
                for child in children:
                    if len(child.children) > 0: # Do not merge nodes that are not leaf nodes
                        continue
                    if child.question_count <= prune_size:
                        assert child.name not in nodes_to_merge
                        nodes_to_merge.append(child.name)

                if len(nodes_to_merge) < 1: # No nodes to merge
                    continue

                # Case 1: all children can be merged, prune the children and map them to the paretnt
                if len(nodes_to_merge) == len(children):
                    for child in children:
                        assert child.name not in mapping
                        assert child.name in graph
                        mapping[child.name] = parent.name
                    parent.children = []
                    parent.question_count = sum([child.question_count for child in children])
                    parents.append(parent.name) # The parent becomes a new leaf node, check if it can be merged
                    continue
                
                # Case 2: not all children can be merged, map the children that can be merged to the first mergeablechild
                new_children = [graph[nodes_to_merge[0]]]
                for child in parent.children:
                    assert child.name not in mapping
                    assert child.name in graph
                    if child.name in nodes_to_merge:
                        if child.name != nodes_to_merge[0]:
                            mapping[child.name] = nodes_to_merge[0]
                            graph[nodes_to_merge[0]].question_count += child.question_count # Update the question count of the first mergeable child
                    else: # Keep the unmergeable children
                        new_children.append(child)
                parent.children = new_children # The parent is not a leaf node so it will not be processed in the next iteration
                
        current_nodes = parents
    
    # Handle the case when a node is map to the parent, then the parent is also mapped to the grandparent, etc.
    check_mapping = True
    while check_mapping:
        check_mapping = False
        for node in mapping.keys():
            if mapping[node] in mapping:
                mapping[node] = mapping[mapping[node]]
                check_mapping = True

    # Remove the merged nodes
    for node in mapping.keys():
        graph.pop(node)
    
    print('After merging: ', len(graph))
    return graph, mapping