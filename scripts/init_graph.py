# scripts/init_graph.py

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KC_tree.io import save_graph, load_graph
from KC_tree.build import convert_to_graph, count_question_count
from KC_tree.prune import prune_graph
from KC_tree.extract_subtree_graph import main as extract_subtree_graph
from config.config import (
    dataset_dir,
    graph_dir
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--prune_size', type=int, default=20, help="Merge nodes with less than prune_size questions")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    prune_size = args.prune_size

    print(f"[INFO] Building graph for dataset: {dataset}")

    # === Stage 1: Build full graph ===
    dep_path = os.path.join(dataset_dir, dataset, 'dependency_mapping.json')
    q_path = os.path.join(dataset_dir, dataset, 'question_info.json')
    graph_out_path = os.path.join(graph_dir, dataset)
    os.makedirs(graph_out_path, exist_ok=True)

    graph = convert_to_graph(dep_path)
    graph = count_question_count(graph, q_path)
    save_graph(graph, graph_out_path)

    # === Stage 2: Prune graph ===
    pruned_graph, param_graph, mapping = prune_graph(graph, prune_size)

    save_graph(pruned_graph, graph_out_path, filename='pruned_knowledge_graph.json')
    save_graph(param_graph, graph_out_path, filename='pruned_parameter_graph.json')
    with open(os.path.join(graph_out_path, 'merged_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f"[DONE] Graphs saved to: {graph_out_path}")

    # === Stage 3: Extract subtree graph ===
    if dataset == 'XES3G5M':
        for node in ['应用题模块', '计数模块','计算模块']:
            extract_subtree_graph(node)
    elif dataset == 'MOOCRadar':
        for node in ['白酒知识', '运算放大器与电路设计','教育理论与实践']:
            extract_subtree_graph(node)

if __name__ == '__main__':
    main()
