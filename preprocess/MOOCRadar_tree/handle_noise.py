import pandas as pd
import json
import os
import ast
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.LLM_tools.prompt import recap_system_prompt, recap_user_prompt
from preprocess.LLM_tools.gpt_call import call_gpt

# Mapping file
level0_mapping_file = './preprocess/MOOCRadar_tree/cluster_summary_mapping_level0.json'
level1_mapping_file = './preprocess/MOOCRadar_tree/cluster_summary_mapping_level1.json'
level2_mapping_file = './preprocess/MOOCRadar_tree/cluster_summary_mapping_level2.json'

with open(level0_mapping_file, 'r') as f:
    level0_mapping = json.load(f)
with open(level1_mapping_file, 'r') as f:
    level1_mapping = json.load(f)
with open(level2_mapping_file, 'r') as f:
    level2_mapping = json.load(f)


def main():
    level_0_file = './preprocess/MOOCRadar_tree/kc_cluster_level0.csv'
    level_1_file = './preprocess/MOOCRadar_tree/kc_cluster_level1.csv'
    level_2_file = './preprocess/MOOCRadar_tree/kc_cluster_level2.csv'

    level0_df = pd.read_csv(level_0_file)
    level1_df = pd.read_csv(level_1_file)
    level2_df = pd.read_csv(level_2_file)

    noise_kc = level2_df[level2_df['cluster'] == -1]['kc'].tolist()

    # kc_question_map = dict()
    
    file_dir = './preprocess/full_data/MOOCRadar'
    problem_file = os.path.join(file_dir, 'problem.json')
    collection = []
    with open(problem_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            collection.append(data)
    p_df = pd.DataFrame(collection)

    # Collection of each level's nodes
    level2_nodes = get_level_nodes(level2_df,2)
    level1_nodes = get_level_nodes(level1_df,1)
    level0_nodes = get_level_nodes(level0_df,0)


    result_list = []
    index = 1
    for kc in noise_kc:
        print(f'Processing {index} out of {len(noise_kc)}')
        index += 1
        result = {
            'level_0':kc,
            'level_1':kc,
            'level_2':kc,
            'root_kc':kc,
            'final_cluster':-1
        }
        # map the kc back to original kc
        kc = map_kc_back(kc)
        
        tmp = p_df[p_df['concepts'].apply(lambda x: kc in x)].reset_index(drop=True)
        assert len(tmp) > 0


        example_question = tmp['detail'][0]
        parsed = ast.literal_eval(example_question)
        question = parsed['content']
        option = parsed['option']
        example_question = question + '\n' + str(option)
        
        recap_finish = False
        max_retry = 3
        category = None

        for level in ['root', 2, 1]: # From general to specific

            if recap_finish:
                break

            retry = 0
            key = 'root_kc' if level == 'root' else f'level_{level}'

            if level == 'root':
                candidate_categories = list(level2_nodes.keys())
            else:
                # Check if the noise is already in this level
                check_level = level0_nodes.keys() if level == 1 else level1_nodes.keys()
                if kc in check_level:
                    recap_finish = True
                    break

                assert category is not None
                target_cluster = category
                # level_df = level0_df if level == 1 else level1_df

                target_mapping = level1_mapping if level == 1 else level2_mapping

                candidate_categories = []
                for k, value in target_mapping.items():
                    if value == target_cluster:
                        candidate_categories.append(k)
                # candidate_categories = get_level_nodes(level_df,level-1,target_cluster)

            system_prompt = recap_system_prompt()
            user_prompt = recap_user_prompt(kc, example_question, candidate_categories)

            while retry < max_retry:
                try:
                    response = call_gpt(system_prompt, user_prompt)
                    breakpoint()
                    response = json.loads(response)
                    category = response['category']
                    if category != 'None':
                        assert category in candidate_categories
                        result[key] = category
                        if level == 'root':
                            result['final_cluster'] = level2_nodes[category]
                    elif category == 'None':
                        recap_finish = True
                    break
                except Exception as e:
                    print(e)
                    retry += 1
            if retry == max_retry:
                recap_finish = True
        result_list.append(result)
    result_df = pd.DataFrame(result_list)
    result_df.to_csv('./preprocess/MOOCRadar_tree/noise_recap_df.csv', index=False)

def get_level_nodes(df,level):
    nodes = dict()
    mapping = level0_mapping if level == 0 else level1_mapping if level == 1 else level2_mapping
    
    clusters = df['cluster'].unique()
    for cluster in clusters:
        if cluster == -1:
            continue
        first_kc = df[df['cluster'] == cluster]['kc'].reset_index(drop=True).iloc[0]
        if first_kc not in mapping.keys():
            assert level != 2
            cluster_name = first_kc
        else:
            cluster_name = mapping[first_kc]
        nodes[cluster_name] = cluster
    return nodes
    

def map_kc_back(kc):
    if kc in level2_mapping.values():
        # find the key
        for key, value in level2_mapping.items():
            if value == kc:
                kc = key
                break
    if kc in level1_mapping.values():
        # find the key
        for key, value in level1_mapping.items():
            if value == kc:
                kc = key
                break
    if kc in level0_mapping.values():
        # find the key
        for key, value in level0_mapping.items():
            if value == kc:
                kc = key
                break
    return kc

if __name__ == '__main__':
    main()