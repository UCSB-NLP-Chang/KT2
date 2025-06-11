from preprocess.LLM_tools.prompt import cluster_summary_system_prompt, cluster_summary_user_prompt
from preprocess.LLM_tools.gpt_call import call_gpt
import pandas as pd
import json
import os
import re

max_retries = 3

def main(current_level=0):
    file_path = f'./preprocess/MOOCRadar_tree/kc_cluster_level{current_level}.csv'
    df = pd.read_csv(file_path)
    kcs = df['kc'].tolist()

    if os.path.exists(f'./preprocess/MOOCRadar_tree/cluster_summary_mapping_level{current_level}.json'):
        with open(f'./preprocess/MOOCRadar_tree/cluster_summary_mapping_level{current_level}.json', 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}

        # Sample one cluster
        all_clusters = df['cluster'].unique()
        progress = 0
        for cluster_id in all_clusters:
            progress += 1
            print(f"Processing cluster {progress} out of {len(all_clusters)}...")
            if cluster_id == -1:
                continue
            kcs = df[df['cluster'] == cluster_id]['kc'].tolist()
            system_prompt = cluster_summary_system_prompt()
            user_prompt = cluster_summary_user_prompt(kcs)
            retries = 0
            while retries < max_retries:
                try:
                    summary = call_gpt(system_prompt, user_prompt)
                    assert_no_punctuation(summary)
                    break
                except Exception as e:
                    print(f"Output contains punctuation: {summary}, retry...")
                    retries += 1
            if retries == max_retries:
                breakpoint() # manually handle
            for kc in kcs:
                mapping[kc] = summary

        # Save the mapping
        with open(f'./preprocess/MOOCRadar_tree/cluster_summary_mapping_level{current_level}.json', 'w') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)

def assert_no_punctuation(text):
    punctuation_pattern = r"[,.!?;:'\"“”‘’、，。？！；：（）《》【】\[\]\(\)【】—…·-]"
    assert not re.search(punctuation_pattern, text), f"Output contains punctuation: {text}"

if __name__ == '__main__':
    main(0)