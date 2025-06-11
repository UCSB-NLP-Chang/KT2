# run initial EM estimation for KT2

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KT2.module.data_cache import initialize_train_uids, get_train_uids
from KC_tree.extract_subtree_graph import main as extract_subtree_graph

from config.config import (
    burn_in_size,
    dataset,
    root_node,
    graph_dir
)

from KT2.EM.EM import main as EM_main
from KT2.KT.KT import main as KT_main

def main():

    # Extract the subtree graph if not exist
    graph_path = os.path.join(graph_dir, dataset)
    if not os.path.exists(os.path.join(graph_path, f'subtree/{root_node}_subtree.json')):
        extract_subtree_graph(root_node)

    initialize_train_uids()
    test_uids, _, train_uids = get_train_uids()

    if burn_in_size > 0:
        uids = train_uids + test_uids
    else:
        uids = train_uids

    print('Total number of EM students:', len(uids))

    EM_main(uids)
    KT_main()

if __name__ == "__main__":
    main()