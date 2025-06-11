import json
import numpy as np
from copy import deepcopy
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import itertools


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KT2.EM.E_Step import e_step
from KT2.EM.M_Step import m_step
from KT2.module.data_cache import initialize_train_uids, get_train_uids, initialize_data, load_data, get_emission_dict
from KT2.module.calibration import calibration
from time import time

from KC_tree.io import (
    save_graph,
    load_graph
)
from config.config import (
    EM_output_dir,
    graph_dir,
    dataset,
    initial_r_diff,
    burn_in_size,
    root_node,
    early_stopping_threshold,
    early_stopping,
    max_step
)

translation_mapping = {
    'Application_Module': '应用题模块',
    'Computation_Module': '计算模块',
    'Counting_Module': '计数模块',
    'Wine_Knowledge': '白酒知识',
    'Circuit_Design': '运算放大器与电路设计',
    'Education_Theory': '教育理论与实践'
}

if root_node in translation_mapping.keys():
    root_name = translation_mapping[root_node]
else:
    root_name = root_node


parameter_graph_path = os.path.join(graph_dir, dataset, 'pruned_parameter_graph.json')
merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')
graph_path = os.path.join(graph_dir, dataset,'subtree')

EM_output_path = os.path.join(EM_output_dir, dataset, f'EM_results-Set{root_node}-burn-in{burn_in_size}')
os.makedirs(EM_output_path, exist_ok=True)

def main(uids):
    '''
    Run EM algorithm
    '''
    update_end = False
    E_upward_time = []
    E_downward_time = []
    calibration_time = []
    M_time = []
    
    graphs = dict()
    single_kc_question_maps = dict()

    initialize_data(has_graph=False)
    datas, _ = load_data()

    _, train_size, _ = get_train_uids()

    with open(parameter_graph_path, 'r') as f:
        parameter_graph = json.load(f)

    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)

    # Load graphs and datas
    for uid in uids:
        # Knowledge State Graph
        graph = load_graph(graph_path, f'{root_node}_subtree.json')
        graphs[uid] = graph
        data_size = train_size[uid]
        datas[uid] = datas[uid][:data_size]

        # Create a single-KC : Questions map for each student
        single_kc_question_map = dict()
        for item in datas[uid]:
            kc = item['kc']
            if kc in mapping:
                kc = mapping[kc]
            if kc not in parameter_graph:
                continue
            if kc not in single_kc_question_map:
                single_kc_question_map[kc] = []
            single_kc_question_map[kc].append(item) # NOTE: duplicate questions should be kept for each KC
        single_kc_question_maps[uid] = single_kc_question_map

    for step in range(max_step):
        print(f'Processing Step {step}')

        parameter_graph_old = deepcopy(parameter_graph) # Keep track the old parameters for early stopping
        phi = parameter_graph[root_name]['phi']
        epsilon = parameter_graph[root_name]['epsilon']
        print('phi:', phi)
        print('epsilon:', epsilon)

        if step == 0:
            r_diff = initial_r_diff # Initial phi_n for easy, medium, hard

        print('r_diff:', r_diff)

        upward_time = 0
        downward_time = 0
        # E-step for each student
        emission_dict = get_emission_dict(reset=True) # Reset emission probability since parameter_graph is updated   
        for uid in uids: 
            graphs[uid], single_upward_time, single_downward_time = e_step(graph=graphs[uid], single_kc_question_map=single_kc_question_maps[uid], parameter_graph=parameter_graph, r_diff=r_diff)
            upward_time += single_upward_time
            downward_time += single_downward_time
        E_upward_time.append(upward_time/len(uids))
        E_downward_time.append(downward_time/len(uids))

        # M-step for all students
        start_time = time()
        r_diff = calibration(uids, graphs, train_size) # calibrate r_diff and save it to parameter_graphoint()

        # Ensure no nan
        for index in range(len(r_diff)):
            if np.isnan(r_diff[index]):
                r_diff[index] = min(initial_r_diff[index], r_diff[index-1]) if index > 0 else initial_r_diff[index]

        calibration_time.append(time()-start_time)
        start_time = time()
        parameter_graph = m_step(graphs, datas, uids, r_diff, parameter_graph, mapping)
        M_time.append(time()-start_time)
       
        # Early stopping
        if step > 0:
            total_diff = 0
            all_nodes = list(parameter_graph.keys())
            total_diff += np.abs(parameter_graph[all_nodes[0]]['phi']-parameter_graph_old[all_nodes[0]]['phi'])
            total_diff += np.abs(parameter_graph[all_nodes[0]]['epsilon']-parameter_graph_old[all_nodes[0]]['epsilon'])
            for node in parameter_graph:
                total_diff += np.abs(parameter_graph[node]['gamma']-parameter_graph_old[node]['gamma'])
            print(f'Step {step} total diff: {total_diff}')
            if early_stopping and total_diff < early_stopping_threshold:
                print(f'Early stopping at step {step}')
                save_results(parameter_graph, 'final', graphs, uids)
                update_end = True
            if step == max_step-1:
                save_results(parameter_graph, 'final', graphs, uids)
                update_end = True

        if update_end:
            break

    print('EM algorithm finished.')

    # Save the time to csv
    # pd.DataFrame({'E_upward_time': E_upward_time, 'E_downward_time': E_downward_time, 'calibration_time': calibration_time, 'M_time': M_time}).to_csv(EM_output_path+'/time.csv', index=False)

    # # plot the time in the same plot
    # plt.plot(E_upward_time, label='E_upward')
    # plt.plot(E_downward_time, label='E_downward')
    # plt.plot(calibration_time, label='calibration')
    # plt.plot(M_time, label='M')
    # plt.legend()
    # plt.savefig(EM_output_path+'/time.png')


def save_results(parameter_graph, step, graphs, uids, EM_output_path=EM_output_path):
    if not os.path.exists(EM_output_path+'/students_graphs'):
        os.makedirs(EM_output_path+'/students_graphs')
    if not os.path.exists(EM_output_path+'/parameter_graphs'):
        os.makedirs(EM_output_path+'/parameter_graphs')

    for uid in uids:
        save_graph(graphs[uid], EM_output_path+'/students_graphs', f'E_step_student_{uid}_step_{step}.json')
        with open(EM_output_path+f'/parameter_graphs/parameter_graph_step_{step}.json', 'w') as f:
            json.dump(parameter_graph, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    initialize_train_uids()
    test_uids, train_size, train_uids = get_train_uids()

    if burn_in_size > 0:
        uids = train_uids + test_uids
    else:
        uids = train_uids

    print('Total number of EM students:', len(uids))
    main(uids)