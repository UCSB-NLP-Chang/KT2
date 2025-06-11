import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from umap import UMAP
import hdbscan
import json
from collections import Counter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_initial_kcs():
    file_dir = './preprocess/full_data/MOOCRadar'
    problem_file = os.path.join(file_dir, 'problem.json')
    collection = []
    with open(problem_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            collection.append(data)
    df = pd.DataFrame(collection)
    # Knowledge Concepts
    all_kcs = []
    for i in df['concepts']:
        all_kcs.extend([item.strip() for item in i])

    # Keep the top 100 most frequent KCs
    kc_counts = Counter(all_kcs)

    kc_count_map = {kc: count for kc, count in kc_counts.items()}

    selected_kcs = []

    for i in df['concepts']:
        if any(kc.strip() in selected_kcs for kc in i):
            continue
        else:
            selected_kcs.append(max(i, key=lambda x: kc_count_map[x.strip()]))
    selected_kcs = list(set(selected_kcs))

    print(f"Initial KC count: {len(selected_kcs)}")
    return selected_kcs

def load_summary_mapping(df, current_level):
    with open(f'./preprocess/MOOCRadar_tree/cluster_summary_mapping_level{current_level-1}.json', 'r') as f:
        mapping = json.load(f)
    df['kc_summary'] = [mapping[kc] if kc in mapping else kc for kc in df['kc']]
    print('total kc_summary count: ', len(df['kc_summary'].tolist()))
    unique_kc = list(set(df['kc_summary'].tolist()))
    print('unique kc_summary count: ', len(unique_kc))
    return unique_kc

def hierarchical_clustering_kcs(
    output_prefix='kc_df_level',
    current_level=0,
    level_params=None
):
    """
    file_dir: path to the directory containing `question_info.jsonl`
    output_prefix: prefix for the output CSV files
    current_level: current level of clustering
    level_params: a list of dicts, each specifying the clustering params for a level
    """

    # Step 1: Load KCs from JSON
    if current_level == 0:
        selected_kcs = load_initial_kcs()
    else:
        file_path = f'./preprocess/MOOCRadar_tree/kc_cluster_level{current_level - 1}.csv'
        df = pd.read_csv(file_path)
        selected_kcs = load_summary_mapping(df, current_level)

    # Prepare KC DataFrame
    kc_df = pd.DataFrame(selected_kcs, columns=['kc'])
    kc_df['kc'] = kc_df['kc'].str.replace('《', '').str.replace('》', '')

    # Load embedding model
    model = SentenceTransformer('BAAI/bge-base-zh')

    # Recursive levels
    current_kc_list = kc_df['kc'].tolist()

    print(f"\n--- Clustering Level {current_level} ---")
    params = level_params[current_level]
    embeddings = model.encode(current_kc_list)
    reducer = UMAP(
        n_neighbors=params.get('n_neighbors', 15),
        n_components=params.get('n_components', 20),
        min_dist=params.get('min_dist', 0.05),
        metric=params.get('umap_metric', 'cosine')
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.get('min_cluster_size', 3),
        min_samples=params.get('min_samples', 3),
        cluster_selection_method=params.get('cluster_selection_method', 'leaf'),
        cluster_selection_epsilon=params.get('cluster_selection_epsilon', 0.01),
        metric=params.get('hdbscan_metric', 'euclidean'),
        prediction_data=True
    )
    labels = clusterer.fit_predict(reduced_embeddings)

    df_out = pd.DataFrame({
        'kc': current_kc_list,
        'cluster': labels
    })
    df_out.to_csv(f"./preprocess/MOOCRadar_tree/{output_prefix}{current_level}.csv", index=False)

    print(f"Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Noise points: {np.sum(labels == -1)}")

    # Prepare next level's KC list using summary placeholder (could replace with real summaries)
    grouped = df_out[df_out['cluster'] != -1].groupby('cluster')['kc'].apply(lambda x: ' / '.join(x))
    current_kc_list = grouped.tolist()

if __name__ == '__main__':
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

    level = 0
    hierarchical_clustering_kcs(
        output_prefix='kc_cluster_level',
        current_level=level,
        level_params=level_params
    )