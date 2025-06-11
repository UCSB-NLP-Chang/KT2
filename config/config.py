import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# KC Graph and Parameter Graph
dataset_dir = config['dataset_dir']
graph_dir = config['graph_dir']

# Output Paths
EM_output_dir = config['EM_output_dir']
KT_output_dir = config['KT_output_dir']

# EM
dataset = config['dataset']
root_node = config['root_node']
burn_in_size = config['burn_in_size']
early_stopping = config['early_stopping']
max_step = config['max_step']
early_stopping_threshold = config['early_stopping_threshold']

# KT
save_intermediate_graph = config['save_intermediate_graph']

# Parameters
initial_r_diff = config['initial_r_diff']
initial_gamma_root = config['initial_gamma_root']
initial_transition = config['initial_transition']
initial_phi = config['initial_phi']
initial_epsilon = config['initial_epsilon']