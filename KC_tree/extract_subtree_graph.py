import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    graph_dir,
    dataset,
    root_node
)
from KC_tree.io import load_graph, save_graph

graph_path = os.path.join(graph_dir, dataset)

translation_mapping = {
    '应用题模块': 'Application_Module',
    '计算模块': 'Computation_Module', 
    '计数模块': 'Counting_Module',
    '白酒知识': 'Wine_Knowledge',
    '运算放大器与电路设计': 'Circuit_Design',
    '教育理论与实践': 'Education_Theory'
}

def main(target_root):
    graph = load_graph(graph_path, 'pruned_knowledge_graph.json')

    if target_root in translation_mapping:
        root_name = translation_mapping[target_root]
    else:
        root_name = target_root

    
    target_subtree_nodes = get_target_subtree_nodes(target_root)
    target_subtree_nodes.append(target_root)

    # Only keep the target node and its subtree
    delete_nodes = []
    for node in graph.keys():
        if node not in target_subtree_nodes:
            delete_nodes.append(node)
    for node in delete_nodes:
        del graph[node]

    # Remove the parent of the target root
    graph[target_root].parents = []
    save_graph(graph, graph_path, f'subtree/{root_name}_subtree.json')

def get_target_subtree_nodes(target_node):
    all_kcs = []
    single_graph = load_graph(graph_path, f'pruned_knowledge_graph.json')
    current_nodes = [single_graph[target_node]]

    while True:
        next_nodes = []
        for node in current_nodes:
            all_kcs.extend([child.name for child in node.children])
            next_nodes.extend([child for child in node.children])
        current_nodes = next_nodes
        if len(current_nodes) == 0:
            break
    all_kcs = list(set(all_kcs))
    return all_kcs

if __name__ == '__main__':
    main(root_node)