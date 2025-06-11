
import json
import os
import pandas as pd
import sys
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.preprocess_config.config import (
    MOOCRadar_dir
)

file_path = {
    'MOOCRadar': "./preprocess/MOOCRadar_tree/kc_tree.csv"
}

'''
Format example:
{
    "拓展思维:几何模块:直线型:几何思想与方法": [
        "拓展思维:几何模块:直线型:几何思想与方法:几何思想平移",
        "拓展思维:几何模块:直线型:几何思想与方法:几何方法整体减空白",
        "拓展思维:几何模块:直线型:几何思想与方法:几何方法差不变",
        "拓展思维:几何模块:直线型:几何思想与方法:分割与剪拼",
        "拓展思维:几何模块:直线型:几何思想与方法:叠加覆盖",
        "拓展思维:几何模块:直线型:几何思想与方法:巧求周长",
        "拓展思维:几何模块:直线型:几何思想与方法:平移法求周长",
        "拓展思维:几何模块:直线型:几何思想与方法:几何思想旋转"
    ],
}
'''
def create_mappings(dataset='MOOCRadar'):
     '''
        - Level_0: orginial kc in dataset
        - Level_1: kc after mapping, which will be used as the leaf node that directly associated with each question
        - Level_2: the parent node of level_1
        - Root_kc: the root node of the tree
        - Final_cluster: the cluster id of the kc
    
    We need to create two dict:
        1. a mapping from level_0 to level_1
        2. a dict that stores the tree structure, following the format example above
     '''
     output_path = os.path.join(MOOCRadar_dir, 'dependency_mapping.json')

     df = pd.read_csv(file_path[dataset])

     kc_mapping = dict()
     graph_dict = dict()

     keep_roots = []

     for _, row in df.iterrows():
        level_0 = row['level_0']
        level_1 = row['level_1']
        level_2 = row['level_2']
        root_kc = row['root_kc']
        final_cluster = row['final_cluster']

        if final_cluster==-1:
            keep_roots.append(root_kc)

        kc_mapping[level_0] = level_1

        level_1_kc = ':'.join([root_kc, level_2, level_1])
        level_2_kc = ':'.join([root_kc, level_2])

        kcs_item = [root_kc, level_2_kc, level_1_kc]
        kcs = [root_kc, level_2, level_1]

        for index in range(2):
            if kcs_item[index] not in graph_dict:
                graph_dict[kcs_item[index]] = []

            if kcs[index+1] == kcs[index]:
                continue
            if kcs_item[index+1] not in graph_dict[kcs_item[index]]:
                graph_dict[kcs_item[index]].append(kcs_item[index+1])
     # remove all kc that has no children
     remove_kcs = []
     for kc in graph_dict:
        if len(graph_dict[kc]) == 0:
            remove_kcs.append(kc)
     for kc in remove_kcs:
        del graph_dict[kc]
     for kc in keep_roots:
        graph_dict[kc] = []
        
     with open(output_path, 'w') as f:
        json.dump(graph_dict, f, indent=4, ensure_ascii=False)
     with open(os.path.join(MOOCRadar_dir, 'initial_mapping.json'), 'w') as f:
        json.dump(kc_mapping, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    create_mappings()