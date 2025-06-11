import numpy as np
import os
import sys
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from KT2.module.utils import emission_probability, transition_probability

def e_step(graph, single_kc_question_map, parameter_graph, r_diff):
    '''
    Upward-Downward algorithm
    '''
    # Upward pass
    current_nodes = [graph[node] for node in graph.keys() if len(graph[node].children) == 0] # Leaf nodes
    start_time = time()

    has_updated = []

    while True:
        next_nodes = []

        for node in current_nodes:
            '''
            NOTE: here we keep track of the upward_beta = [beta_0, beta_1] for both mastery = 0 and 1 to enable calculating beta tilde.
            Later, given the mastery value, we can use node.upward_beta[node.mastery] to get the correct beta of each node.
            '''
            if node.children != []:
                should_update = True
                # If any child node has not been updated, we should not update the current node
                for child in node.children:
                    if child not in has_updated:
                        next_nodes.append(node)
                        should_update = False
                        break
                if not should_update:
                    continue

            new_beta = upward_pass(node, single_kc_question_map, parameter_graph, r_diff)
            # Normalize the upward_beta for each node across k=0 and k=1
            divide = sum(new_beta)
            if divide == 0:
                breakpoint()
                divide = 1e-6
            new_beta[0] = new_beta[0]/divide
            new_beta[1] = new_beta[1]/divide
            node.upward_beta = new_beta
            assert len(node.parents) <= 1 # Ensure only one parent for each node
            if len(node.parents) == 1:
                parent = node.parents[0]
                next_nodes.append(parent)
            has_updated.append(node)
        
        # Stop at the root nodes
        if len(next_nodes) == 0:
            # print('All nodes have been processed in Downward-pass.')
            break
        current_nodes = list(set(next_nodes))
    E_upward_time = time()-start_time

    # Downward pass
    current_nodes = [graph[node] for node in graph.keys() if len(graph[node].parents) == 0] # Root nodes
    start_time = time()

    while True:
        next_nodes = []
        for node in current_nodes:
            '''
            NOTE: similiar to the upward pass, we keep track of the downward_alpha = [gamma_0, gamma_1] for both mastery = 0 and 1.
            '''
            
            new_alpha = downward_pass(node, parameter_graph)        
            # Normalize the downward_alpha for each node across k=0 and k=1
            divide = sum(new_alpha)
            if divide == 0:
                breakpoint()
                divide = 1e-6
            new_alpha[0] = new_alpha[0]/divide
            new_alpha[1] = new_alpha[1]/divide
            node.downward_alpha = new_alpha
            next_nodes.extend(node.children)

        if len(next_nodes) == 0:
            # print('All nodes have been processed in Upward-pass.')
            break

        current_nodes = list(set(next_nodes))
    E_downward_time = time()-start_time
    # Calculate posterior probability
    for node_name in graph.keys():
        node = graph[node_name]
        node.posterior1, node.posterior2, node.posterior3 = calculate_posterior(node, parameter_graph)

    # print('Posterior probability calculated.')
    return graph, E_upward_time, E_downward_time

def upward_pass(node, single_kc_question_map, parameter_graph, r_diff):
    '''
    Upward algorithm: compute beta_j(k) at each KC node j
    
    1. Base case: at leaf nodes, compute the emission probability  only; beta_j(k) = Sigma_{t} P(Q_t|K_j)
    2. Recursive case: beta_j(k) = Pi_{t} P(Q_t|K_j) * Pi_{l} beta_tilde_{l,j}(k)
    '''
    upward_beta = [0,0]
    # Base case
    if len(node.children) == 0: # Leaf node
        for mastery in [0, 1]:
            upward_beta[mastery] = emission_probability(node, mastery, single_kc_question_map, parameter_graph, r_diff)
    else:
        # Emision probability at the current node * product of beta_tilde at all children
        children = node.children
        for mastery in [0, 1]:
            all_beta_tilde = []
            for child in children:
                all_beta_tilde.append(beta_tilde(child, mastery, parameter_graph))
            emission = emission_probability(node, mastery, single_kc_question_map, parameter_graph, r_diff)
            upward_beta[mastery] = emission * np.prod(all_beta_tilde)
    return upward_beta

def beta_tilde(node, parent_mastery, parameter_graph):
    '''
    beta_tilde_{l,j}(k) = Sum_{k_l} beta_l(k_l) * p(K_l = k_l | K_j = k)
    '''
    # Sum over all possible mastery values
    result = 0
    for mastery in [0, 1]:
        result += transition_probability(node, mastery, parent_mastery, parameter_graph) * node.upward_beta[mastery]
    if result == 0:
        breakpoint()
    return result


def downward_pass(node, parameter_graph):
    '''
    Downward algorithm: computebet gamma_j(k) at each KC node j
    '''
    downward_alpha = [0,0]
    # Base case
    if len(node.parents) == 0: # Root node
        # Return the prior probability of p(K_j = k)
        for mastery in [0, 1]:
            downward_alpha[mastery] = transition_probability(node, mastery, None, parameter_graph)
    else:
        parent = node.parents[0]
        for mastery in [0, 1]:
            result = 0
            # Sum over all possible mastery values of the parent node
            for parent_mastery in [0, 1]:
                result += transition_probability(node, mastery, parent_mastery, parameter_graph) * parent.downward_alpha[parent_mastery] * parent.upward_beta[parent_mastery]/beta_tilde(node, parent_mastery, parameter_graph)
            downward_alpha[mastery] = result
    return downward_alpha

def calculate_posterior(node, parameter_graph):
    # P(K_ji =1 | Q_i = q_i)
    posterior1 = node.downward_alpha[1] * node.upward_beta[1] / sum([node.downward_alpha[1] * node.upward_beta[1], node.downward_alpha[0] * node.upward_beta[0]])

    if np.isnan(posterior1):
        breakpoint()
    # print('Posterior1', posterior1)
    
    if len(node.parents) == 0: # Root node
        return posterior1, None, None

    # P(K_ji = 1, K_P(j)i != 1 | Q_i = q_i)
    posterior2 = calcualte_posterior23(node, 1, 0, parameter_graph)

    # P(K_ji = 0, K_P(j)i != 1 | Q_i = q_i)
    posterior3 = calcualte_posterior23(node, 0, 0, parameter_graph)

    return posterior1, posterior2, posterior3

def calcualte_posterior23(node, k_j, k_pj,parameter_graph):
    '''
    P(K_ji = k_j, K_P(j)i = k_pj | Q_i = q_i)
    '''
    parent = node.parents[0]

    result = parent.downward_alpha[k_pj] * node.upward_beta[k_j] * transition_probability(node, k_j, k_pj, parameter_graph)*parent.upward_beta[k_pj]/beta_tilde(node, k_pj, parameter_graph)
    divide = 0
    for mastery in [0, 1]:
        for parent_mastery in [0, 1]:
            divide += parent.downward_alpha[parent_mastery] * node.upward_beta[mastery] * transition_probability(node, mastery, parent_mastery, parameter_graph)*parent.upward_beta[parent_mastery]/beta_tilde(node, parent_mastery, parameter_graph)

    if divide == 0:
        breakpoint()

    result /= divide
    return result