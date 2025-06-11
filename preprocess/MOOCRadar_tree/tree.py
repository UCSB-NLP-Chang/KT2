import pandas as pd
import json
from io import StringIO
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    all_level_dfs = []
    all_level_mappings = []
    for level in range(3):
        file_path = f'./preprocess/MOOCRadar_tree/kc_cluster_level{level}.csv'
        df = pd.read_csv(file_path)
        all_level_dfs.append(df)

        mapping_path = f'./preprocess/MOOCRadar_tree/cluster_summary_mapping_level{level}.json'
        mapping = json.load(open(mapping_path))
        all_level_mappings.append(mapping)

    initial_df = all_level_dfs[0]
    level_0_kc = initial_df['kc'].tolist()
    level_1_kc = []
    for kc in level_0_kc:
        level_1_kc.append(all_level_mappings[0][kc] if kc in all_level_mappings[0] else kc)
    
    level_2_kc = []
    for kc in level_1_kc:
        level_2_kc.append(all_level_mappings[1][kc] if kc in all_level_mappings[1] else kc)
    
    root_kc = []
    final_clusters = []
    for kc in level_2_kc:
        new_kc = all_level_mappings[2][kc] if kc in all_level_mappings[2] else kc
        root_kc.append(new_kc)
        df = all_level_dfs[2][all_level_dfs[2]['kc'] == kc]
        final_clusters.append(df['cluster'].tolist()[0])

    new_df = pd.DataFrame({
        'level_0': level_0_kc,
        'level_1': level_1_kc,
        'level_2': level_2_kc,
        'root_kc': root_kc,
        'final_cluster': final_clusters,
    })

    print('Total number of kcs: ', len(new_df))

    new_df = combine_noise_df(new_df)
    print('Total number of not-noise kcs: ', len(new_df[new_df['final_cluster']!=-1]))

    new_df.to_csv('./preprocess/MOOCRadar_tree/kc_tree.csv', index=False)


    # Visualize the tree
    tree_text = build_tree_text(new_df)
    with open("kc_tree_from_dataframe_all.txt", "w", encoding="utf-8") as f:
        f.write(tree_text)

def combine_noise_df(df):
    noise_in_df = df[df['final_cluster'] == -1].reset_index(drop=True)

    recap_df = pd.read_csv('./preprocess/MOOCRadar_tree/noise_recap_df.csv')
    recap_df = recap_df[['level_0', 'level_1', 'level_2', 'root_kc', 'final_cluster']]

    noise_df = []
    for _, row in noise_in_df.iterrows():
        single_df = dict()
        level2_kc = row['level_2']
        tmp = recap_df[recap_df['level_0'] == level2_kc]
        assert len(tmp) == 1
        single_df['level_0'] = row['level_0']
        for key in ['level_1', 'level_2', 'root_kc']:
            if row[key] == level2_kc:
                single_df[key] = tmp[key].values[0]
            else:
                single_df[key] = row[key]
        
        single_df['final_cluster'] = tmp['final_cluster'].values[0]
        single_df = pd.DataFrame([single_df])
        noise_df.append(single_df)
    noise_df = pd.concat(noise_df)
    df = df[df['final_cluster']!=-1]
    df = pd.concat([df, noise_df])
    return df

def build_tree_text(df):
    output = StringIO()
    for root in sorted(df['root_kc'].dropna().unique()):
        print(root, file=output)
        df_root = df[df['root_kc'] == root]
        for level2 in sorted(df_root['level_2'].dropna().unique()):
            if level2 == root:
                continue
            print(f"├─ {level2}", file=output)
            df_level2 = df_root[df_root['level_2'] == level2]
            level1_kcs = sorted(df_level2['level_1'].dropna().unique())
            for idx, level1 in enumerate(level1_kcs):
                if level1 == level2:
                    continue
                prefix = "│   ├─" if idx < len(level1_kcs) - 1 else "│   └─"
                print(f"{prefix} {level1}", file=output)
    return output.getvalue()

if __name__ == '__main__':
    main()
