import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.MOOCRadar_tree.level_clustering import hierarchical_clustering_kcs
from preprocess.MOOCRadar_tree.summary import main as summary_main
from preprocess.MOOCRadar_tree.handle_noise import main as handle_noise
from preprocess.MOOCRadar_tree.tree import main as build_tree
from preprocess.MOOCRadar_tree.create_graph_mapping import create_mappings

def main():

    level_params=[
        {   # first level clustering
            'n_neighbors': 30,
            'n_components': 20,
            'min_dist': 0.05,
            'min_cluster_size': 3,
            'min_samples': 3,
            'cluster_selection_method': 'leaf',
            'cluster_selection_epsilon': 0.01
        },
        {   # second level clustering
            'n_neighbors': 15,
            'n_components': 10,
            'min_dist': 0.1,
            'min_cluster_size': 2,
            'min_samples': 2,
            'cluster_selection_method': 'eom',
            'cluster_selection_epsilon': 0.02
        },
        {   # third level clustering
            'n_neighbors': 50,                
            'n_components': 15,               
            'min_dist': 0.2,                  
            'min_cluster_size': 8,            
            'min_samples': 2,                 
            'cluster_selection_method': 'eom',
            'cluster_selection_epsilon': 0.05,
            'hdbscan_metric': 'euclidean',    
        }
    ]

    for level in range(0, 3):
        hierarchical_clustering_kcs(
            output_prefix='kc_cluster_level',
            current_level=level,
            level_params=level_params
        )
        summary_main(level)
    handle_noise()
    build_tree()
    create_mappings()


if __name__ == '__main__':
    main()