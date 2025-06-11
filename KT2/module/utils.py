import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import (
    initial_gamma_root
)
from KT2.module.data_cache import get_emission_dict


def emission_probability(node, mastery, single_kc_question_map, parameter_graph, r_diff):
    '''
    P(Q_t|K_j)
    '''
    if node.name not in single_kc_question_map:
        return 1.0
    score = []
    emission_dict = get_emission_dict()

    # Emission probability
    for item in single_kc_question_map[node.name]:
        if mastery == 1:
            if item['question'] in emission_dict.keys():
                phi_n = emission_dict[item['question']]
            else:
                difficulty_label = item['difficulty']                
                difficulty_mapping = {"easy": 0, "medium": 1, "hard": 2}
                # NOTE: handle case when the question has no difficulty rating
                phi = parameter_graph[node.name]['phi'] 
                try:
                    phi_n = r_diff[difficulty_mapping[difficulty_label]]
                    assert phi_n > 0 and phi_n < 1
                except Exception as e:
                    print(e)
                    print('Error found.')
                    breakpoint()
                emission_dict[item['question']] = phi_n
            score.append(phi_n if item['response'] == 1 else 1-phi_n)
        else:
            score.append(parameter_graph[node.name]['epsilon'] if item['response'] == 1 else 1-parameter_graph[node.name]['epsilon'])
            
    mul_score = np.prod(score)
    if mul_score == 0 or np.isnan(mul_score): # handle python float precision issue when extreme small value is regarded as 0
        mul_score = 1e-6
    return mul_score

def transition_probability(node, mastery, parent_mastery, parameter_graph):
    '''
    P(K_j = k | K_P[j] = k_P[j])
    '''
    root_prior = parameter_graph[node.name]['gamma_root']
    if root_prior is None:
        root_prior = initial_gamma_root
    if parent_mastery == None: # Root node, prior probability of p(K_j = k)
        if mastery == 1:
            return root_prior
        else:
            return 1-root_prior
    else:
        if parent_mastery == 1:
            return 1 if mastery == 1 else 0
        else:
            return parameter_graph[node.name]['gamma'] if mastery == 1 else 1-parameter_graph[node.name]['gamma']