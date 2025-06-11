# cache.py
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    graph_dir,
    dataset_dir,
    EM_output_dir,
    burn_in_size,
    dataset,
    root_node
)
from KC_tree.io import load_graph

global_datas = {}
global_graphs = {}


recording_path = os.path.join(dataset_dir, dataset, 'recordings.jsonl')
question_path = os.path.join(dataset_dir, dataset, 'question_info.json')

graph_path = os.path.join(graph_dir, dataset)
test_path = os.path.join(f'{dataset_dir}/{dataset}/subtree_split/selected_classroom_students_{root_node}0.65.txt')

merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')

translation_mapping = {
    'Application_Module': '应用题模块',
    'Computation_Module': '计算模块',
    'Counting_Module': '计数模块',
    'Wine_Knowledge': '白酒知识',
    'Circuit_Design': '运算放大器与电路设计',
    'Education_Theory': '教育理论与实践'
}


def get_target_subtree_nodes(root_node):
    if root_node in translation_mapping.keys():
        root_node = translation_mapping[root_node]
    
    all_kcs = []
    single_graph = load_graph(graph_path, f'pruned_knowledge_graph.json')
    current_nodes = [single_graph[root_node]]

    while True:
        next_nodes = []
        for node in current_nodes:
            all_kcs.extend([child.name for child in node.children])
            next_nodes.extend([child for child in node.children])
        current_nodes = next_nodes
        if len(current_nodes) == 0:
            break
    all_kcs = list(set(all_kcs))

    with open(os.path.join(graph_path, f'subtree/{root_node}_subtree_nodes.json'), 'w') as f:
        json.dump(all_kcs, f, indent=4, ensure_ascii=False)
    return all_kcs

def initialize_data(has_graph=True):

    global global_datas
    global global_graphs

    if global_datas and global_graphs:
        return global_datas, global_graphs

    uids = train_uids + test_uids
    uids = list(set(uids))

    print("Loading data...") 

    with open(question_path, 'r') as f:
        questions = json.load(f)

    target_subtree_nodes = get_target_subtree_nodes(root_node)

    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)


    for uid in uids:
        # Knowledge State Graph
        if has_graph:
            EM_output_path = os.path.join(EM_output_dir, dataset, f'EM_results-Set{root_node}-burn-in{burn_in_size}')
            if not os.path.exists(EM_output_path+'/students_graphs'+f'/E_step_student_{uid}_step_final.json'):
                single_graph = load_graph(graph_path, f'{root_node}_subtree.json')
            else:
                single_graph = load_graph(EM_output_path+'/students_graphs', f'E_step_student_{uid}_step_final.json')
            global_graphs[uid] = single_graph
        
    # random.seed(42)
    with open(recording_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = str(data['student_id'])

            if uid not in uids:
                continue

            all_questions = []
            for index in range(len(data['exercises_logs'])):
                tmp = dict()
                que = data['exercises_logs'][index]
                response = int(data['is_corrects'][index])
                if response == -1:
                    breakpoint()
                assert que in questions.keys()
                question = questions[que]
                tmp['question'] = question['content']
                tmp['response'] = response
                tmp['kc'] = question['kc']
                tmp['difficulty'] = question['difficulty']

                kc = question['kc']
                if kc in mapping.keys():
                    kc = mapping[kc]
                if kc not in target_subtree_nodes: # Filter for both train and test, keep only the subtree questions
                    continue
                all_questions.append(tmp)
            if uid in train_uids:
                print('Filtering out questions not in the train subtree:',train_size[uid], len(all_questions))
                train_size[uid] = len(all_questions)
            else:
                print('Filtering out questions not in the test subtree:',train_size[uid], len(all_questions))
            global_datas[uid] = all_questions

def load_data():
    return global_datas, global_graphs

train_uids = []
test_uids = []
train_size = {}

def initialize_train_uids():
    global train_uids
    global train_size
    global test_uids

    if len(train_size) > 0:
        return global_datas, global_graphs

    train_uids = []
    with open(test_path, 'r') as f:
        test_uids = f.readlines()
        test_uids = [uid.strip() for uid in test_uids]
    
    print(f'Number of train students: {len(train_uids)}')
    print(f'Number of test students: {len(test_uids)}')

    with open(recording_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = str(data['student_id'])
            total_length = len(data['exercises_logs'])
            if uid in test_uids:
                train_size[uid] = burn_in_size
            elif uid in train_uids:
                train_size[uid] = total_length

def get_train_uids():
    return test_uids, train_size, train_uids

emission_dict = {}

def get_emission_dict(reset=False):
    global emission_dict
    if reset:
        emission_dict = {}
    return emission_dict