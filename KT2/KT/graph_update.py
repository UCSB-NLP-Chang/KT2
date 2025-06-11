
import os
import copy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KT2.module.data_cache import load_data, get_train_uids, get_emission_dict
from EM.E_Step import upward_pass, downward_pass, calculate_posterior, e_step
from EM.M_Step import m_step
from KT2.module.calibration import calibration
from config.config import (
    burn_in_size
)
max_step = 1


def update_graph(graph, index, parameter_graph, theta, uids, current_uid, mapping):
    '''
    Update the graph based on the new data
    '''
    graphs = dict()
    datas = dict()

    datas, graphs = load_data()
    test_uids, train_size, train_uids = get_train_uids()


    assert uids == test_uids

    original_data = copy.deepcopy(datas[current_uid])

    data_size = train_size[current_uid]
    datas[current_uid] = datas[current_uid][:data_size+index]
                  
    graphs[current_uid] = graph # the graph to be updated
    new_data = datas[current_uid][-1]
    new_data_kc = new_data['kc']
    if new_data_kc in mapping:
        new_data_kc = mapping[new_data_kc]
    

    # Create a single-KC : Questions map for current student
    single_kc_question_map = dict()
    for item in datas[current_uid]:
        kc = item['kc']
        if kc in mapping:
            kc = mapping[kc]
        if kc not in single_kc_question_map:
            single_kc_question_map[kc] = []
        single_kc_question_map[kc].append(item) # NOTE: duplicate questions should be kept for each KC
    
    
    # Update the graph for the current student only
    for step in range(max_step):
        emission_dict = get_emission_dict(reset=True)
        if burn_in_size == 0 and graphs[current_uid][new_data_kc].upward_beta == [0,0]:
            graphs[current_uid] = e_step(graphs[current_uid], single_kc_question_map, parameter_graph, theta)[0]
        else:
            graphs[current_uid] = update_e_step(graphs[current_uid], single_kc_question_map, parameter_graph, theta, new_data_kc)
    
        # M-step
        train_size[current_uid] += index
        if burn_in_size == 0:
            theta = calibration([current_uid]+train_uids, graphs, train_size) # calibrate theta and save it to parameter_graph
        else:
            theta = calibration(uids+train_uids, graphs, train_size) # calibrate theta and save it to parameter_graph

        train_size[current_uid] -= index

        if burn_in_size == 0:
            parameter_graph = m_step(graphs, datas, [current_uid]+train_uids, theta, parameter_graph, mapping)
        else:
            parameter_graph = m_step(graphs, datas, uids+train_uids, theta, parameter_graph, mapping)

    datas[current_uid] = original_data # restore the original data
    return graphs[current_uid], parameter_graph, theta


def update_e_step(graph, single_kc_question_map, parameter_graph, theta, new_data_kc):
    '''
    Upward-Downward algorithm
    '''
    # Upward pass
    current_nodes = [graph[new_data_kc]] # NOTE: here we only consider the new data as the leaf node

    while True:
        next_nodes = []

        for node in current_nodes:
            '''
            NOTE: here we keep track of the upward_beta = [beta_0, beta_1] for both mastery = 0 and 1 to enable calculating beta tilde.
            Later, given the mastery value, we can use node.upward_beta[node.mastery] to get the correct beta of each node.
            '''

            new_beta = upward_pass(node, single_kc_question_map, parameter_graph, theta)
            divide = sum(new_beta)
            new_beta[0] = new_beta[0]/divide
            new_beta[1] = new_beta[1]/divide
            node.upward_beta = new_beta
            assert len(node.parents) <= 1 # Ensure only one parent for each node
            if len(node.parents) == 1:
                parent = node.parents[0]
                next_nodes.append(parent)
        # Stop at the root nodes
        if len(next_nodes) == 0:
            # print('All nodes have been processed in Downward-pass.')
            break
        current_nodes = next_nodes
    # Downward pass
    current_nodes = [graph[node] for node in graph.keys() if len(graph[node].parents) == 0] # Root nodes
    while True:
        next_nodes = []
        for node in current_nodes:
            '''
            NOTE: similiar to the upward pass, we keep track of the downward_gamma = [alpha_0, alpha_1] for both mastery = 0 and 1.
            '''
            new_gamma = downward_pass(node, parameter_graph)
            divide = sum(new_gamma)
            new_gamma[0] = new_gamma[0]/divide
            new_gamma[1] = new_gamma[1]/divide
            node.downward_gamma = new_gamma
            next_nodes.extend(node.children)

        if len(next_nodes) == 0:
            # print('All nodes have been processed in Upward-pass.')
            break

        current_nodes = next_nodes
    # Calculate posterior probability
    for node_name in graph.keys():
        node = graph[node_name]
        node.posterior1, node.posterior2, node.posterior3 = calculate_posterior(node, parameter_graph)

    return graph