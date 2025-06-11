import json
import os
import sys
import numpy as np
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    graph_dir,
    dataset
)

from KT2.module.data_cache import load_data

merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')


def calibration(uids, graphs, train_size):


    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Knowledge State Graph
    datas, _ = load_data()

    statistics = {}

    correct_difficulty = []
    incorrect_difficulty = []

    correct_posterior1 = []
    incorrect_posterior1 = []

    correct_question = []
    incorrect_question = []


    for uid in uids:
        data = copy.deepcopy(datas[uid])
        data_size = train_size[uid]
        data = data[:data_size]

        student_graph = graphs[uid]
        
        for item in data:
            question = item['question']
            response = item['response']
            kc = item['kc']
            if kc in mapping:
                kc = mapping[kc]
            if kc not in student_graph:
                print('KC Not Found:', kc)
                continue

            
            diffculty_label = item['difficulty']
            if response == 1:
                correct_difficulty.append(diffculty_label)
                correct_posterior1.append(student_graph[kc].posterior1)
                correct_question.append(question)
            else:
                incorrect_difficulty.append(diffculty_label)
                incorrect_posterior1.append(student_graph[kc].posterior1)
                incorrect_question.append(question)

            if diffculty_label not in statistics:
                statistics[diffculty_label] = []
            statistics[diffculty_label].append(response)


    r_diff1, r_diff2, r_diff3 = calibrate_r_diff_closed_form(correct_difficulty, incorrect_difficulty, correct_posterior1, incorrect_posterior1)

    return [r_diff1, r_diff2, r_diff3]


def calibrate_r_diff_closed_form(correct_difficulty, incorrect_difficulty, correct_posterior1, incorrect_posterior1):
    difficulty_mapping = {"easy": 0, "medium": 1, "hard": 2}

    correct_posterior1 = np.array(correct_posterior1)
    incorrect_posterior1 = np.array(incorrect_posterior1)

    X_correct = np.array([difficulty_mapping[label] for label in correct_difficulty])
    X_incorrect = np.array([difficulty_mapping[label] for label in incorrect_difficulty])

    r_diff = [0, 0, 0]

    for i in range(3):
        mask_correct = (X_correct == i)
        mask_incorrect = (X_incorrect == i)

        sum_correct = np.sum(correct_posterior1[mask_correct])
        sum_incorrect = np.sum(incorrect_posterior1[mask_incorrect])

        r_diff[i] = sum_correct/(sum_correct+sum_incorrect)
    
    # Handle python float precision issue & ensure r_diff0 > r_diff1 > r_diff2
    r_diff[0] = np.clip(r_diff[0], 1e-6, 1-1e-6)
    r_diff[1] = np.clip(r_diff[1], 1e-6, 1-1e-6)
    r_diff[2] = np.clip(r_diff[2], 1e-6, 1-1e-6)
    r_diff[0] = np.min([1-1e-6, r_diff[0]])
    r_diff[1] = np.min([r_diff[0], r_diff[1]])
    r_diff[2] = np.min([r_diff[1], r_diff[2]])

    return r_diff
