import numpy as np
def m_step(graphs, datas, uids, r_diff, parameter_graph, mapping):
    '''
    Update gamma, r_diff, epsilon
    '''
    new_r_diff = [0,0]
    new_epsilon = [0,0]
    all_gammas = dict()
    all_root_gammas = dict()
    
    for uid in uids:
        graph = graphs[uid]
        data = datas[uid]

        _r_diff, _epsilon = update_emission_assumption(graph, data, mapping)
        new_r_diff[0] += _r_diff[0]
        new_r_diff[1] += _r_diff[1]
        new_epsilon[0] += _epsilon[0]
        new_epsilon[1] += _epsilon[1]

        all_nodes = list(graph.keys())
    
        for node in all_nodes:
            if len(graph[node].parents) == 0:
                if node not in all_root_gammas:
                    all_root_gammas[node] = update_root_node_gamma(graph[node])
                else:
                    all_root_gammas[node][0] += update_root_node_gamma(graph[node])[0]
                    all_root_gammas[node][1] += update_root_node_gamma(graph[node])[1]
                continue
            if node not in all_gammas:
                all_gammas[node] = update_node_gamma(graph[node])
            else:
                all_gammas[node][0] += update_node_gamma(graph[node])[0]
                all_gammas[node][1] += update_node_gamma(graph[node])[1]
    new_r_diff = new_r_diff[0]/(new_r_diff[0] + new_r_diff[1])

    if new_r_diff >= 1 or np.isnan(new_r_diff): # handle python float precision issue, i.e., regard extremly small value in denominator as 0
        new_r_diff = 1-1e-6

    new_epsilon = new_epsilon[0]/(new_epsilon[0] + new_epsilon[1])
    if new_epsilon <= 0 or np.isnan(new_epsilon): # handle python float precision issue
        new_epsilon = 1e-6
    if new_epsilon > 0.3:
        new_epsilon = 0.3
    if new_epsilon > r_diff[2]: # Ensure epsilon is not greater than phi[hard]
        new_epsilon = r_diff[2]-1e-6

    for node in all_nodes:
        parameter_graph[node]['phi'] = new_r_diff
        parameter_graph[node]['epsilon'] = new_epsilon
        parameter_graph[node]['r_diff'] = r_diff
        if len(graph[node].parents) == 0:
            continue

        if all_gammas[node][0] + all_gammas[node][1] == 0:
            parameter_graph[node]['gamma'] = 1e-6
        else:
            parameter_graph[node]['gamma'] = all_gammas[node][0] / (all_gammas[node][0] + all_gammas[node][1])

        # Clip gamma between 1e-6 and 1-1e-6
        parameter_graph[node]['gamma'] = np.clip(parameter_graph[node]['gamma'], 1e-6, 1-1e-6)

    for node in all_root_gammas:
        if all_root_gammas[node][0] + all_root_gammas[node][1] == 0:
            print('Cuurent root node', node, 'get prior as 0.')
            parameter_graph[node]['gamma_root'] = 1e-6
            breakpoint()
        else:
            parameter_graph[node]['gamma_root'] = all_root_gammas[node][0] / (all_root_gammas[node][0] + all_root_gammas[node][1])
    return parameter_graph

def update_emission_assumption(graph, data, mapping):
    '''
    Update emission assumption (r_diff, epsilon)

    posterior2(q_i) = p_old(K_ji = 1, K_P[j]i != 1 | Q_i = q_i)
    posterior3(q_i) = p_old(K_ji = 0, K_P[j]i != 1 | Q_i = q_i)

    phi = posterior1(q_i=1) / [ posterior1(q_i=1) + posterior1(q_i=0) ]
    epsilon = (1-posterior1(q_i=1)) / [ (1-posterior1(q_i=1)) + (1-posterior1(q_i=0)) ]
            = (1-posterior1(q_i=1)) / [ posterior1(q_i=0) - posterior1(q_i=1) ]
    '''
    correct_question_temp = [item['kc'] for item in data if item['response'] == 1]
    incorrect_question_temp = [item['kc'] for item in data if item['response'] == 0]

    # NOTE: mapping is a dictionary that maps the original kc to the merged kc
    correct_question = []
    incorrect_question = []
    for item in correct_question_temp:
        if item in mapping:
            correct_question.append(mapping[item])
        else:
            correct_question.append(item)
    for item in incorrect_question_temp:
        if item in mapping:
            incorrect_question.append(mapping[item])
        else:
            incorrect_question.append(item)
    # Filter out the questions that are not in the graph
    correct_question = [item for item in correct_question if item in graph]
    incorrect_question = [item for item in incorrect_question if item in graph]
    
    def _compute_posterior1(collections, if_r_diff=True):
        posterior1 = 0
        for item in collections:
            if if_r_diff:
                posterior1 += graph[item].posterior1
            else:
                posterior1 += (1-graph[item].posterior1)
        return posterior1
    
    posterior1_correct_r_diff = _compute_posterior1(correct_question)
    posterior1_incorrect_r_diff = _compute_posterior1(incorrect_question)
    posterior1_correct_epsilon = _compute_posterior1(correct_question, if_r_diff=False)
    posterior1_incorrect_epsilon = _compute_posterior1(incorrect_question, if_r_diff=False)

    r_diff = [posterior1_correct_r_diff, posterior1_incorrect_r_diff]
    epsilon = [posterior1_correct_epsilon, posterior1_incorrect_epsilon] 

    return r_diff, epsilon


def update_node_gamma(node):
    '''
    Update gamma

    posterior1(q_i) = p_old(K_M(t)i = 1 | Q_i = q_i)

    gamma = posterior2(q_i) / (posterior2(q_i) + posterior3(q_i))
    '''
    posterior2 = node.posterior2
    posterior3 = node.posterior3

    return [posterior2, posterior3]

def update_root_node_gamma(node):
    '''
    Update gamma for root node
    '''
    posterior1 = node.posterior1

    return [posterior1, 1-posterior1]
