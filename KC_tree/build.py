import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    dataset_dir,
    graph_dir
)

from KC_tree.node import KCNode

def convert_to_graph(file):
    # load a json file where the graphs are stored in a dictionary
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create nodes for all keys and their children
    all_nodes = set()
    for key in data.keys():
        if ':' in key:
            key = key.split(':')[-1]
        all_nodes.add(key)
    for value in data.values():
        for child in value:
            if ':' in child:
                child = child.split(':')[-1]
            all_nodes.add(child)
    
    graph = {name: KCNode(name) for name in all_nodes}

    # add the children to the graph
    for name, children in data.items():
        if ':' in name:
            name = name.split(':')[-1]
        for child in children:
            if ':' in child:
                child = child.split(':')[-1]
            if graph[child] not in graph[name].children:
                graph[name].add_child(graph[child])
                if graph[name] not in graph[child].parents:
                    graph[child].add_parent(graph[name])

    return graph


def count_question_count(graph, path, dataset=None, mapping_path=None):
    '''
    Count the number of questions for each node
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _, value in data.items():
        kc = value['kc']
        graph[kc].question_count += 1
    return graph