import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KC_tree.node import KCNode

def save_graph(graph, path, filename="knowledge_graph.json"):
    """Save the graph dictionary to a JSON file"""
    json_data = {name: node.to_dict() for name, node in graph.items()}

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+'/'+filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        

def load_graph(path, filename="knowledge_graph.json"):
    """Load the graph from a JSON file"""
    with open(path+'/'+filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recreate the nodes
    graph = {name: KCNode(name) for name in data}

    # Rebuild relationships
    for name, node_data in data.items():
        graph[name].question_count = node_data["question_count"]

        # Posterior
        graph[name].downward_alpha = node_data["downward_alpha"]
        graph[name].upward_beta = node_data["upward_beta"]
        graph[name].posterior1 = node_data["posterior1"]
        graph[name].posterior2 = node_data["posterior2"]
        graph[name].posterior3 = node_data["posterior3"]

        # Children and parents
        for child_name in node_data["children"]:
            if child_name in graph:  # Ensure the child exists
                graph[name].add_child(graph[child_name])
            else:
                print(f"Child node {child_name} not found in graph")
        for parent_name in node_data["parents"]:
            if parent_name in graph:  # Ensure the parent exists
                graph[name].add_parent(graph[parent_name])
            else:
                print(f"Parent node {parent_name} not found in graph")

    return graph